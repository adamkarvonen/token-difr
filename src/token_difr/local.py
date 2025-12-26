"""Local verification using vLLM."""

from __future__ import annotations

import gc
from typing import Any

import torch
from tqdm import tqdm

from token_difr.common import (
    SamplingMethod,
    TokenMetrics,
    TokenSequence,
    _as_list,
    _get_probs,
    _verify_vllm_gumbel_max,
    compute_metrics_summary,
)


def _prompt_logprobs_to_tensor(
    prompt_logprobs: list[dict[int, Any] | None],
    start_idx: int,
    n_tokens: int,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    """Convert a slice of prompt_logprobs into a dense full-vocabulary tensor."""
    slice_rows = prompt_logprobs[start_idx : start_idx + n_tokens]
    if len(slice_rows) != n_tokens:
        raise ValueError(f"Expected {n_tokens} prompt logprob rows, got {len(slice_rows)}")

    logits = torch.full((n_tokens, vocab_size), float("-inf"), device=device)

    for j, row in enumerate(slice_rows):
        if not row:
            continue

        token_ids = torch.tensor([int(tok_id) for tok_id in row.keys()], device=device, dtype=torch.long)
        logprobs = torch.tensor([float(val.logprob) for val in row.values()], device=device)
        logits[j].scatter_(0, token_ids, logprobs)

    return logits


@torch.inference_mode()
def verify_outputs(
    outputs: list[TokenSequence],
    model_name: str,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    max_model_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    vllm_kwargs: dict[str, Any] | None = None,
    sampling_method: SamplingMethod = SamplingMethod.VLLM_GUMBEL_MAX,
    gpu_memory_utilization: float = 0.7,
    verbose: bool = True,
) -> list[list[TokenMetrics]]:
    """
    Verify LLM outputs using Gumbel-Max sampling verification (requires vLLM).

    This function takes token sequences (prompt + generated output) and verifies
    whether the outputs could have been produced by the specified model using
    the given sampling parameters. Uses vLLM to load the model and compute logits.

    Args:
        outputs: List of TokenSequence objects containing prompt and output token IDs.
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        temperature: Sampling temperature used during generation. Required.
        top_k: Top-k sampling parameter. Required.
        top_p: Top-p (nucleus) sampling parameter. Required.
        seed: Random seed used during generation. Required.
        max_model_len: Maximum model context length. If None, auto-computed from outputs.
        dtype: Model dtype (e.g., torch.bfloat16, torch.float16). Default: torch.bfloat16.
        vllm_kwargs: Additional keyword arguments to pass to vLLM's LLM constructor.
            Useful for quantization (e.g., {"quantization": "fp8"}) or other settings.
        sampling_method: The sampling method to verify against. Default: VLLM_GUMBEL_MAX.
        gpu_memory_utilization: Fraction of GPU memory to use. Default: 0.7.
        verbose: Whether to show progress and print a summary. Default: True.

    Returns:
        List of lists of TokenMetrics, one per token in each output sequence.
        Each TokenMetrics contains:
            - exact_match: Whether the token matches under verification
            - prob: Probability of the actual token
            - margin: Margin between max and actual token scores
            - logit_rank: Rank of actual token by logit value
            - gumbel_rank: Rank of actual token by Gumbel score

    Example:
        >>> from token_difr import verify_outputs, TokenSequence
        >>> outputs = [
        ...     TokenSequence(
        ...         prompt_token_ids=[128000, 2323, 374],
        ...         output_token_ids=[264, 1296, 13]
        ...     )
        ... ]
        >>> results = verify_outputs(
        ...     outputs,
        ...     model_name="meta-llama/Llama-3.1-8B-Instruct",
        ...     temperature=1.0,
        ...     seed=42,
        ... )
    """
    # Import vLLM here to avoid import errors when only using Tinker backend
    import os

    os.environ.setdefault("VLLM_USE_V1", "1")  # Enable vLLM v1 features

    try:
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt
    except ImportError as e:
        raise ImportError(
            "vLLM backend requires the vllm package. Install with: pip install token-difr[all]"
        ) from e

    if sampling_method != SamplingMethod.VLLM_GUMBEL_MAX:
        raise ValueError(
            f"Unsupported sampling method: {sampling_method}. Only VLLM_GUMBEL_MAX is currently supported."
        )

    vllm_kwargs = vllm_kwargs or {}

    # Auto-compute max_model_len if not provided
    if max_model_len is None:
        max_seq_len = max(len(o.prompt_token_ids) + len(o.output_token_ids) for o in outputs)
        max_model_len = max_seq_len * 2  # Add buffer

    model = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype=dtype,  # type: ignore[arg-type]
        gpu_memory_utilization=gpu_memory_utilization,
        logprobs_mode="raw_logits",
        max_logprobs=top_k,
        **vllm_kwargs,
    )

    tokenizer = model.get_tokenizer()
    vocab_size = len(tokenizer)

    prompt_logprob_params = SamplingParams(
        prompt_logprobs=top_k,
        max_tokens=1,
        logprobs=0,
        detokenize=False,
    )

    device_for_inputs = torch.device("cuda")

    # Build all prompts for batched generation
    verify_prompts: list[TokensPrompt] = []
    request_metadata: list[tuple[int, int, list[int]]] = []

    iterator = enumerate(outputs)
    if verbose:
        iterator = enumerate(tqdm(outputs, desc="Preparing verification prompts"))

    for i, req in iterator:
        prompt_token_ids: list[int] = _as_list(req.prompt_token_ids)
        gen_ids: list[int] = _as_list(req.output_token_ids)

        if len(gen_ids) == 0:
            continue

        seq_concat: list[int] = prompt_token_ids + gen_ids
        verify_prompts.append({"prompt_token_ids": seq_concat})
        request_metadata.append((i, len(prompt_token_ids), gen_ids))

    # Batched generation
    if verbose:
        print(f"Running batched verification for {len(verify_prompts)} sequences...")
    verify_results = model.generate(verify_prompts, sampling_params=prompt_logprob_params)

    # Compute metrics from results
    all_token_metrics: list[list[TokenMetrics]] = [[] for _ in outputs]

    metadata_iterator = request_metadata
    if verbose:
        metadata_iterator = tqdm(request_metadata, desc="Computing verification metrics")

    for batch_idx, (orig_idx, prompt_len, gen_ids) in enumerate(metadata_iterator):
        verify_req = verify_results[batch_idx]
        gen_len = len(gen_ids)

        if verify_req.prompt_logprobs is None:
            raise ValueError("vLLM did not return prompt_logprobs; enable prompt_logprobs in SamplingParams.")

        logits_JV = _prompt_logprobs_to_tensor(
            verify_req.prompt_logprobs,
            start_idx=prompt_len,
            n_tokens=gen_len,
            device=device_for_inputs,
            vocab_size=vocab_size,
        )

        J = logits_JV.shape[0]
        gold_col_idx_J = torch.as_tensor(gen_ids, device=device_for_inputs, dtype=torch.long)

        top_k_tensor_J = torch.full((J,), top_k, device=logits_JV.device, dtype=torch.long)
        top_p_tensor_J = torch.full((J,), top_p, device=logits_JV.device, dtype=logits_JV.dtype)

        logits_JV = logits_JV.float()
        probs_JV = _get_probs(logits_JV, temperature, top_k_tensor_J, top_p_tensor_J)

        row_idx_J = torch.arange(J, device=device_for_inputs)
        gold_logits_J = logits_JV[row_idx_J, gold_col_idx_J]
        logit_ranks_J = (logits_JV > gold_logits_J.unsqueeze(1)).sum(dim=1).float()

        probs_gold_J = probs_JV.gather(1, gold_col_idx_J.view(-1, 1)).squeeze(1)

        if sampling_method == SamplingMethod.VLLM_GUMBEL_MAX:
            pred_ids_J, gumbel_ranks_J, margins_J = _verify_vllm_gumbel_max(
                temperature=temperature,
                seed=seed,
                logits_JV=logits_JV,
                probs_JV=probs_JV,
                gold_col_idx_J=gold_col_idx_J,
                top_k_tensor_J=top_k_tensor_J,
                top_p_tensor_J=top_p_tensor_J,
            )
        else:
            raise ValueError(
                f"Unsupported sampling method: {sampling_method}. Supported methods: {SamplingMethod.VLLM_GUMBEL_MAX}"
            )

        seq_token_metrics: list[TokenMetrics] = []
        for j in range(J):
            actual_id = int(gen_ids[j])

            token_metrics = TokenMetrics(
                exact_match=bool(int(pred_ids_J[j]) == actual_id),
                prob=float(probs_gold_J[j].item()),
                margin=float(margins_J[j].item()),
                logit_rank=float(logit_ranks_J[j].item()),
                gumbel_rank=float(gumbel_ranks_J[j].item()),
            )
            seq_token_metrics.append(token_metrics)

        all_token_metrics[orig_idx] = seq_token_metrics

    del model
    torch.cuda.empty_cache()
    gc.collect()

    if verbose:
        compute_metrics_summary(all_token_metrics, verbose=True)

    return all_token_metrics
