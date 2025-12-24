"""API-based verification using Tinker and Fireworks backends."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, Literal

import torch
from openai import OpenAI
from openai.types.completion_choice import Logprobs
from tqdm import tqdm

from token_difr.common import (
    TokenMetrics,
    TokenSequence,
    _as_list,
    compute_metrics_summary,
)

# Type alias for a single position's top-k logprobs: list of (token_id, logprob) tuples
PositionLogprobs = list[tuple[int, float]] | None


@dataclass
class SparseLogprobs:
    """Compact representation of top-k logprobs for a sequence.

    This stores only the top-k (token_id, logprob) pairs per position,
    avoiding the memory cost of full vocab-size tensors.

    Attributes:
        index: Index of this sequence in the original outputs list.
        gen_ids: The generated token IDs being verified.
        logprobs: Per-position sparse logprobs. Each entry is either None
            or a list of (token_id, logprob) tuples for the top-k tokens.
    """

    index: int
    gen_ids: list[int]
    logprobs: list[PositionLogprobs]


def _sparse_logprobs_to_tensor(
    sparse_logprobs: list[PositionLogprobs],
    n_tokens: int,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    """Convert sparse logprobs into a dense full-vocabulary tensor.

    Args:
        sparse_logprobs: Per-position sparse logprobs. Each entry is either None
            or a list of (token_id, logprob) tuples for the top-k tokens.
        n_tokens: Expected number of tokens.
        device: Torch device.
        vocab_size: Vocabulary size for the dense tensor.

    Returns:
        Tensor of shape (n_tokens, vocab_size) with logprobs filled in.
    """
    if len(sparse_logprobs) != n_tokens:
        raise ValueError(f"Expected {n_tokens} logprob rows, got {len(sparse_logprobs)}")

    logits = torch.full((n_tokens, vocab_size), float("-inf"), device=device)

    for j, row in enumerate(sparse_logprobs):
        if row is None:
            continue

        token_ids = torch.tensor([tok_id for tok_id, _ in row], device=device, dtype=torch.long)
        logprobs = torch.tensor([logprob for _, logprob in row], device=device)
        logits[j].scatter_(0, token_ids, logprobs)

    return logits


def _fetch_fireworks_logprobs(
    prompt_token_ids: list[int],
    gen_ids: list[int],
    client: OpenAI,
    model: str,
    tokenizer,
    topk_logprobs: int,
) -> Logprobs | None:
    """Fetch logprobs from Fireworks API (blocking call for use with ThreadPoolExecutor)."""
    prompt_text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    output_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    response = client.completions.create(
        model=model,
        prompt=prompt_text + output_text,
        max_tokens=1,
        logprobs=topk_logprobs,
        echo=True,
    )

    return response.choices[0].logprobs


def _iter_tinker_logprobs(
    request_data: list[tuple[int, list[int], list[int]]],
    sampling_client,
    topk_logprobs: int,
    verbose: bool,
) -> Iterator[SparseLogprobs]:
    """Submit Tinker requests and yield SparseLogprobs as results complete."""
    import tinker

    # Submit all requests (they return futures)
    futures = []
    for i, prompt_token_ids, gen_ids in request_data:
        full_sequence = prompt_token_ids + gen_ids
        full_prompt = tinker.ModelInput.from_ints(full_sequence)

        future = sampling_client.sample(
            prompt=full_prompt,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            num_samples=1,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk_logprobs,
        )
        futures.append((i, prompt_token_ids, gen_ids, future))

    # Yield results with progress bar
    iterator = tqdm(futures, desc="Verifying via tinker API") if verbose else futures
    for i, prompt_token_ids, gen_ids, future in iterator:
        logprob_result = future.result()
        prompt_len = len(prompt_token_ids)
        gen_len = len(gen_ids)

        # Extract just the slice we need (still sparse, no tensor conversion)
        sparse_logprobs = logprob_result.topk_prompt_logprobs[prompt_len : prompt_len + gen_len]
        yield SparseLogprobs(index=i, gen_ids=gen_ids, logprobs=sparse_logprobs)


def _fireworks_to_sparse_logprobs(
    top_logprobs: list[dict[str, float] | None],
    start_idx: int,
    n_tokens: int,
    tokenizer,
) -> list[PositionLogprobs]:
    """Convert Fireworks logprobs to sparse format matching Tinker's output.

    Returns per-position sparse logprobs where each entry is either None
    or a list of (token_id, logprob) tuples.
    """
    slice_rows = top_logprobs[start_idx : start_idx + n_tokens]
    if len(slice_rows) != n_tokens:
        raise ValueError(f"Expected {n_tokens} logprob rows, got {len(slice_rows)}")

    result: list[PositionLogprobs] = []
    for row in slice_rows:
        if row is None:
            result.append(None)
            continue

        token_logprobs: list[tuple[int, float]] = []
        for token_str, logprob in row.items():
            token_ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(token_ids) == 1:
                token_logprobs.append((token_ids[0], logprob))
        result.append(token_logprobs if token_logprobs else None)

    return result


def _iter_fireworks_logprobs(
    request_data: list[tuple[int, list[int], list[int]]],
    client: OpenAI,
    model: str,
    tokenizer,
    topk_logprobs: int,
    verbose: bool,
) -> Iterator[SparseLogprobs]:
    """Submit Fireworks requests and yield SparseLogprobs as results complete."""
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for i, prompt_token_ids, gen_ids in request_data:
            future = executor.submit(
                _fetch_fireworks_logprobs,
                prompt_token_ids=prompt_token_ids,
                gen_ids=gen_ids,
                client=client,
                model=model,
                tokenizer=tokenizer,
                topk_logprobs=topk_logprobs,
            )
            futures.append((i, prompt_token_ids, gen_ids, future))

        # Yield results with progress bar
        iterator = tqdm(futures, desc="Verifying via fireworks API") if verbose else futures
        for i, prompt_token_ids, gen_ids, future in iterator:
            logprobs_data = future.result()
            prompt_len = len(prompt_token_ids)
            gen_len = len(gen_ids)

            # +1 for the empty token Fireworks prepends
            start_idx = prompt_len + 1

            sparse_logprobs = _fireworks_to_sparse_logprobs(
                logprobs_data.top_logprobs,
                start_idx=start_idx,
                n_tokens=gen_len,
                tokenizer=tokenizer,
            )
            yield SparseLogprobs(index=i, gen_ids=gen_ids, logprobs=sparse_logprobs)


def _compute_verification_metrics_from_logprobs(
    logprobs_JV: torch.Tensor,
    gen_ids: list[int],
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    device: torch.device,
) -> list[TokenMetrics]:
    """Compute verification metrics from log probabilities (not raw logits).

    Currently only supports greedy verification (temperature=0). The signature
    includes all sampling parameters for future expansion.

    For greedy verification:
    - exact_match: True if claimed token is the argmax of logprobs
    - prob: probability of the claimed token
    - margin: logprob(top1) - logprob(claimed_token), 0 if exact match
    - logit_rank: rank of claimed token by logprob (0 = highest)
    - gumbel_rank: same as logit_rank for greedy (placeholder for future)
    """
    # Keep parameters for future use
    _ = temperature, top_k, top_p, seed

    J = logprobs_JV.shape[0]
    gold_col_idx_J = torch.as_tensor(gen_ids, device=device, dtype=torch.long)

    # Convert log probs to probs
    probs_JV = torch.exp(logprobs_JV.float())

    row_idx_J = torch.arange(J, device=device)
    gold_logprobs_J = logprobs_JV[row_idx_J, gold_col_idx_J]

    # Compute rank based on log probs (higher logprob = better, so rank 0 = best)
    logit_ranks_J = (logprobs_JV > gold_logprobs_J.unsqueeze(1)).sum(dim=1).float()

    probs_gold_J = probs_JV.gather(1, gold_col_idx_J.view(-1, 1)).squeeze(1)

    # Greedy verification: predicted token is argmax of logprobs
    pred_ids_J = logprobs_JV.argmax(dim=-1)

    # Margin: logprob(top1) - logprob(claimed)
    max_logprobs_J = logprobs_JV.max(dim=-1).values
    margins_J = max_logprobs_J - gold_logprobs_J

    seq_token_metrics: list[TokenMetrics] = []
    for j in range(J):
        actual_id = int(gen_ids[j])
        token_metrics = TokenMetrics(
            exact_match=bool(int(pred_ids_J[j]) == actual_id),
            prob=float(probs_gold_J[j].item()),
            margin=float(margins_J[j].item()),
            logit_rank=float(logit_ranks_J[j].item()),
            gumbel_rank=float(logit_ranks_J[j].item()),  # Same as logit_rank for greedy
        )
        seq_token_metrics.append(token_metrics)

    return seq_token_metrics


@torch.inference_mode()
def verify_outputs_api(
    outputs: list[TokenSequence],
    backend: Literal["tinker", "fireworks"],
    vocab_size: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    # Backend-specific args
    sampling_client=None,  # For Tinker
    client: OpenAI | None = None,  # For Fireworks
    model: str | None = None,  # For Fireworks
    tokenizer=None,  # For Fireworks
    topk_logprobs: int = 20,
    verbose: bool = True,
) -> list[list[TokenMetrics]]:
    """
    Verify LLM outputs using API-based verification.

    This function takes token sequences (prompt + generated output) and verifies
    whether the outputs could have been produced by the specified model using
    the given sampling parameters. Requests are submitted in parallel for efficiency.

    Args:
        outputs: List of TokenSequence objects containing prompt and output token IDs.
        backend: Which backend to use ("tinker" or "fireworks").
        vocab_size: The vocabulary size of the model.
        temperature: Sampling temperature used during generation. Required.
        top_k: Top-k sampling parameter. Required.
        top_p: Top-p (nucleus) sampling parameter. Required.
        seed: Random seed used during generation. Required.
        sampling_client: A Tinker SamplingClient (required for backend="tinker").
        client: An OpenAI client configured for Fireworks (required for backend="fireworks").
        model: The model name for Fireworks (required for backend="fireworks").
        tokenizer: HuggingFace tokenizer (required for backend="fireworks").
        topk_logprobs: Number of top logprobs to request. Default: 20.
        verbose: Whether to show progress and print a summary. Default: True.

    Returns:
        List of lists of TokenMetrics, one per token in each output sequence.
    """
    device = torch.device("cpu")

    all_token_metrics: list[list[TokenMetrics]] = [[] for _ in outputs]

    # Prepare request data
    request_data: list[tuple[int, list[int], list[int]]] = []  # (index, prompt_token_ids, gen_ids)
    for i, req in enumerate(outputs):
        prompt_token_ids: list[int] = _as_list(req.prompt_token_ids)
        gen_ids: list[int] = _as_list(req.output_token_ids)
        if len(gen_ids) == 0:
            continue
        request_data.append((i, prompt_token_ids, gen_ids))

    if len(request_data) == 0:
        return all_token_metrics

    # Get logprobs iterator based on backend
    # Both iterators yield SparseLogprobs - compact representation of top-k logprobs
    if backend == "tinker":
        if sampling_client is None:
            raise ValueError("sampling_client is required for tinker backend")
        logprobs_iter = _iter_tinker_logprobs(
            request_data=request_data,
            sampling_client=sampling_client,
            topk_logprobs=topk_logprobs,
            verbose=verbose,
        )
    elif backend == "fireworks":
        if client is None or model is None or tokenizer is None:
            raise ValueError("client, model, and tokenizer are required for fireworks backend")
        logprobs_iter = _iter_fireworks_logprobs(
            request_data=request_data,
            client=client,
            model=model,
            tokenizer=tokenizer,
            topk_logprobs=topk_logprobs,
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Process all results with unified loop
    # Materialize tensor only when needed to minimize memory usage
    for result in logprobs_iter:
        logprobs_JV = _sparse_logprobs_to_tensor(
            sparse_logprobs=result.logprobs,
            n_tokens=len(result.gen_ids),
            device=device,
            vocab_size=vocab_size,
        )
        seq_token_metrics = _compute_verification_metrics_from_logprobs(
            logprobs_JV=logprobs_JV,
            gen_ids=result.gen_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            device=device,
        )
        all_token_metrics[result.index] = seq_token_metrics

    if verbose:
        summary = compute_metrics_summary(all_token_metrics)
        print("Verification Summary:")
        print(f"  Total tokens: {summary['total_tokens']}")
        print(f"  Exact match rate: {summary['exact_match_rate']:.2%}")
        print(f"  Average probability: {summary['avg_prob']:.4f}")
        print(f"  Average margin: {summary['avg_margin']:.4f} ({summary['infinite_margin_rate']:.2%} infinite)")

    return all_token_metrics
