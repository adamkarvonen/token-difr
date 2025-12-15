"""Token verification using Gumbel-Max sampling."""

from __future__ import annotations

import os

# Enable vLLM v1 features (required for prompt logprobs)
os.environ.setdefault("VLLM_USE_V1", "1")

import gc
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


class SamplingMethod(Enum):
    """Supported sampling methods for verification."""

    VLLM_GUMBEL_MAX = "vllm_gumbel_max"


@dataclass
class TokenSequence:
    """A prompt and its generated output as token IDs."""

    prompt_token_ids: list[int]
    output_token_ids: list[int]


@dataclass
class TokenMetrics:
    """Verification metrics for a single token."""

    exact_match: bool
    prob: float
    margin: float
    logit_rank: float
    gumbel_rank: float


def _exponential_to_gumbel(random_exponentials: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Convert exponential noise E ~ Exp(1) to Gumbel noise G = -log(E)."""
    return -torch.log(random_exponentials.clamp(min=epsilon))


def _apply_top_k_only(
    logits: torch.Tensor,
    k: torch.Tensor,
) -> torch.Tensor:
    """Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    The logits tensor may be updated in-place.
    """
    assert len(logits.shape) == 2
    assert k.shape[0] == logits.shape[0], f"k.shape: {k.shape}, logits.shape: {logits.shape}"

    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = int(k.max().item())
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    logits.masked_fill_(logits < top_k_mask, -float("inf"))
    return logits


def _apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.
    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits
        return _apply_top_k_only(logits, k)

    assert len(logits.shape) == 2

    if k is not None:
        assert k.shape[0] == logits.shape[0], f"k.shape: {k.shape}, logits.shape: {logits.shape}"
    if p is not None:
        assert p.shape[0] == logits.shape[0], f"p.shape: {p.shape}, logits.shape: {logits.shape}"

    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

    if k is not None and (k > 0).all():
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))

    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))

    logits = logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)
    return logits


def _keep_one_token(scores: torch.Tensor, tok_idx: torch.Tensor) -> torch.Tensor:
    """Keep exactly one token per row along the last dimension."""
    assert tok_idx.shape == scores.shape[:-1], (
        f"tok_idx.shape {tok_idx.shape} must match scores.shape[:-1] {scores.shape[:-1]}"
    )
    out = torch.full_like(scores, float("-inf"))
    idx = tok_idx.unsqueeze(-1)
    values = torch.gather(scores, dim=-1, index=idx)
    out.scatter_(dim=-1, index=idx, src=values)
    return out


def _get_probs(logits: torch.Tensor, temperature: float, top_k: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Compute probabilities from logits with temperature and top-k/top-p filtering."""
    assert len(logits.shape) == 2, f"Expected 2D logits, got shape {logits.shape}"

    if temperature > 0.0:
        x = logits / max(temperature, 1e-8)
    else:
        idx = torch.argmax(logits, dim=-1)
        x = _keep_one_token(logits, idx)

    x = _apply_top_k_top_p(x, top_k, top_p)
    probs = torch.nn.functional.softmax(x, dim=-1, dtype=torch.float32)
    return probs


def _compute_margin_batch(
    logits_JV: torch.Tensor,
    random_exponentials_JV: torch.Tensor,
    neg_inf_mask_JV: torch.Tensor,
    temperature: float,
    gold_idx_J: torch.Tensor,
) -> torch.Tensor:
    """Compute max - gold margins for a batch where gold_idx_J indexes logits_JV."""
    assert logits_JV.dim() == 2, f"Expected [J, V] logits, got {logits_JV.shape}"

    random_gumbels_JV = _exponential_to_gumbel(random_exponentials_JV.float(), epsilon=0)
    noised_logits_JV = logits_JV + (random_gumbels_JV * temperature)
    noised_logits_JV[neg_inf_mask_JV] = float("-inf")

    J = logits_JV.shape[0]
    max_idx_J = noised_logits_JV.argmax(dim=-1)
    row_J = torch.arange(J, device=logits_JV.device)
    max_vals_J = noised_logits_JV[row_J, max_idx_J]
    gold_vals_J = noised_logits_JV[row_J, gold_idx_J]
    logit_diff_J = max_vals_J - gold_vals_J

    return logit_diff_J


def _as_list(x) -> list[int]:
    if isinstance(x, torch.Tensor):
        return x.tolist()
    if isinstance(x, tuple):
        return list(x)
    return list(x)


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


def _verify_vllm_gumbel_max(
    temperature: float,
    seed: int,
    logits_JV: torch.Tensor,
    probs_JV: torch.Tensor,
    gold_col_idx_J: torch.Tensor,
    top_k_tensor_J: torch.Tensor,
    top_p_tensor_J: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify the outputs against vLLM's Gumbel-Max sampling."""
    filtered_logits_JV = logits_JV.clone()

    if temperature > 0.0:
        filtered_logits_JV = filtered_logits_JV / max(temperature, 1e-8)

    filtered_logits_JV = _apply_top_k_top_p(filtered_logits_JV, top_k_tensor_J, top_p_tensor_J)
    neg_inf_mask_JV = ~torch.isfinite(filtered_logits_JV)

    generator = torch.Generator(device=logits_JV.device)

    J = logits_JV.shape[0]
    row_idx_J = torch.arange(J, device=logits_JV.device)

    generator.manual_seed(seed)
    exponential_rows = []
    for _ in range(J):
        exp_v = torch.empty_like(probs_JV[0])
        exp_v.exponential_(generator=generator)
        exponential_rows.append(exp_v)
    random_exponentials_JV = torch.stack(exponential_rows, dim=0)

    gumbel_max_scores_JV = probs_JV / random_exponentials_JV
    pred_ids_J = gumbel_max_scores_JV.argmax(dim=-1)

    gold_filtered_J = ~torch.isfinite(filtered_logits_JV[row_idx_J, gold_col_idx_J])

    gold_gumbel_scores_J = gumbel_max_scores_JV[row_idx_J, gold_col_idx_J]
    gumbel_ranks_J = torch.full((J,), float("inf"), device=logits_JV.device)
    valid_mask_J = ~gold_filtered_J
    if valid_mask_J.any():
        higher_scores_counts = (
            gumbel_max_scores_JV[valid_mask_J] > gold_gumbel_scores_J[valid_mask_J].unsqueeze(1)
        ).sum(dim=1)
        gumbel_ranks_J[valid_mask_J] = higher_scores_counts.float()

    margins_J = _compute_margin_batch(
        logits_JV,
        random_exponentials_JV,
        neg_inf_mask_JV=neg_inf_mask_JV,
        temperature=temperature,
        gold_idx_J=gold_col_idx_J,
    )

    return pred_ids_J, gumbel_ranks_J, margins_J


@torch.inference_mode()
def verify_outputs(
    outputs: list[TokenSequence],
    model_name: str,
    *,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int = 42,
    max_model_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    vllm_kwargs: dict[str, Any] | None = None,
    sampling_method: SamplingMethod = SamplingMethod.VLLM_GUMBEL_MAX,
    gpu_memory_utilization: float = 0.7,
    show_progress: bool = True,
) -> list[list[TokenMetrics]]:
    """
    Verify LLM outputs using Gumbel-Max sampling verification.

    This function takes token sequences (prompt + generated output) and verifies
    whether the outputs could have been produced by the specified model using
    the given sampling parameters.

    Args:
        outputs: List of TokenSequence objects containing prompt and output token IDs.
        model_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct").
        temperature: Sampling temperature used during generation. Default: 1.0.
        top_k: Top-k sampling parameter. Default: 50.
        top_p: Top-p (nucleus) sampling parameter. Default: 0.95.
        seed: Random seed used during generation. Default: 42.
        max_model_len: Maximum model context length. If None, auto-computed from outputs.
        dtype: Model dtype (e.g., torch.bfloat16, torch.float16). Default: torch.bfloat16.
        vllm_kwargs: Additional keyword arguments to pass to vLLM's LLM constructor.
            Useful for quantization (e.g., {"quantization": "fp8"}) or other settings.
        sampling_method: The sampling method to verify against. Default: VLLM_GUMBEL_MAX.
        gpu_memory_utilization: Fraction of GPU memory to use. Default: 0.7.
        show_progress: Whether to show progress bars. Default: True.

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
    if show_progress:
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
    if show_progress:
        print(f"Running batched verification for {len(verify_prompts)} sequences...")
    verify_results = model.generate(verify_prompts, sampling_params=prompt_logprob_params)

    # Compute metrics from results
    all_token_metrics: list[list[TokenMetrics]] = [[] for _ in outputs]

    metadata_iterator = request_metadata
    if show_progress:
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

    return all_token_metrics
