"""API-based verification using Tinker and Fireworks backends."""

import asyncio
from dataclasses import dataclass
from typing import Iterator

import torch
from openai import AsyncOpenAI
from tqdm import tqdm

from token_difr.common import (
    TokenMetrics,
    TokenSequence,
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


async def _fetch_fireworks_logprobs(
    prompt_token_ids: list[int],
    gen_ids: list[int],
    client: AsyncOpenAI,
    model: str,
    topk_logprobs: int,
):
    """Fetch logprobs from Fireworks API."""
    # Pass token IDs directly to avoid tokenization mismatches
    # Use logprobs=True with extra_body to get content format with token_ids
    response = await client.completions.create(
        model=model,
        prompt=prompt_token_ids + gen_ids,
        max_tokens=1,
        echo=True,
        logprobs=True,
        extra_body={"top_logprobs": topk_logprobs},
    )

    return response.choices[0].logprobs.content


def _iter_tinker_logprobs(
    outputs: list[TokenSequence],
    sampling_client,
    topk_logprobs: int,
    verbose: bool,
) -> Iterator[SparseLogprobs]:
    """Submit Tinker requests and yield SparseLogprobs as results complete."""
    import tinker

    # Submit all requests (they return futures)
    futures = []
    for i, req in enumerate(outputs):
        if len(req.output_token_ids) == 0:
            continue
        full_sequence = list(req.prompt_token_ids) + list(req.output_token_ids)
        full_prompt = tinker.ModelInput.from_ints(full_sequence)

        future = sampling_client.sample(
            prompt=full_prompt,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            num_samples=1,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk_logprobs,
        )
        futures.append((i, req, future))

    # Yield results with progress bar
    iterator = tqdm(futures, desc="Verifying via tinker API") if verbose else futures
    for i, req, future in iterator:
        logprob_result = future.result()
        prompt_len = len(req.prompt_token_ids)
        gen_len = len(req.output_token_ids)

        # Extract just the slice we need (still sparse, no tensor conversion)
        sparse_logprobs = logprob_result.topk_prompt_logprobs[prompt_len : prompt_len + gen_len]
        yield SparseLogprobs(index=i, gen_ids=list(req.output_token_ids), logprobs=sparse_logprobs)


def _fireworks_to_sparse_logprobs(
    content: list,
    start_idx: int,
    n_tokens: int,
) -> list[PositionLogprobs]:
    """Convert Fireworks logprobs content to sparse format.

    Args:
        content: List of logprob entries from Fireworks API (each has token_id, top_logprobs).
        start_idx: Index to start slicing from.
        n_tokens: Number of tokens to extract.

    Returns per-position sparse logprobs where each entry is either None
    or a list of (token_id, logprob) tuples.
    """
    slice_rows = content[start_idx : start_idx + n_tokens]
    if len(slice_rows) != n_tokens:
        raise ValueError(f"Expected {n_tokens} logprob rows, got {len(slice_rows)}")

    result: list[PositionLogprobs] = []
    for row in slice_rows:
        if row is None or "top_logprobs" not in row:
            result.append(None)
            continue

        # Extract (token_id, logprob) tuples directly - no encoding needed
        token_logprobs = [(entry["token_id"], entry["logprob"]) for entry in row["top_logprobs"]]
        result.append(token_logprobs if token_logprobs else None)

    return result


async def _fetch_all_fireworks_logprobs(
    outputs: list[TokenSequence],
    client: AsyncOpenAI,
    model: str,
    topk_logprobs: int,
    verbose: bool,
    concurrency: int = 10,
) -> list[SparseLogprobs]:
    """Fetch all Fireworks logprobs concurrently and return as list."""
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_one(i: int, req: TokenSequence) -> SparseLogprobs:
        async with semaphore:
            prompt_token_ids = list(req.prompt_token_ids)
            gen_ids = list(req.output_token_ids)
            content = await _fetch_fireworks_logprobs(
                prompt_token_ids=prompt_token_ids,
                gen_ids=gen_ids,
                client=client,
                model=model,
                topk_logprobs=topk_logprobs,
            )

            sparse_logprobs = _fireworks_to_sparse_logprobs(
                content=content,
                start_idx=len(prompt_token_ids),
                n_tokens=len(gen_ids),
            )
            return SparseLogprobs(index=i, gen_ids=gen_ids, logprobs=sparse_logprobs)

    tasks = [fetch_one(i, req) for i, req in enumerate(outputs) if len(req.output_token_ids) > 0]

    if verbose:
        results = []
        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Verifying via fireworks API"):
            results.append(await coro)
        return results
    else:
        return await asyncio.gather(*tasks)


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


def _process_results_to_metrics(
    results: list[SparseLogprobs],
    n_outputs: int,
    vocab_size: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    verbose: bool,
) -> list[list[TokenMetrics]]:
    """Shared logic: convert SparseLogprobs to TokenMetrics.

    Args:
        results: List of SparseLogprobs from either backend.
        n_outputs: Number of output sequences (for pre-allocating results).
        vocab_size: Vocabulary size for dense tensor conversion.
        temperature: Sampling temperature used during generation.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        seed: Random seed used during generation.
        verbose: Whether to print a summary.

    Returns:
        List of lists of TokenMetrics, one per token in each output sequence.
    """
    device = torch.device("cpu")
    all_token_metrics: list[list[TokenMetrics]] = [[] for _ in range(n_outputs)]

    for result in results:
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
        compute_metrics_summary(all_token_metrics, verbose=True)

    return all_token_metrics


@torch.inference_mode()
def verify_outputs_tinker(
    outputs: list[TokenSequence],
    vocab_size: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    sampling_client,
    topk_logprobs: int = 20,
    verbose: bool = True,
) -> list[list[TokenMetrics]]:
    """
    Verify LLM outputs using Tinker API (synchronous).

    This function takes token sequences (prompt + generated output) and verifies
    whether the outputs could have been produced by the specified model using
    the given sampling parameters.

    Args:
        outputs: List of TokenSequence objects containing prompt and output token IDs.
        vocab_size: The vocabulary size of the model.
        temperature: Sampling temperature used during generation.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        seed: Random seed used during generation.
        sampling_client: A Tinker SamplingClient.
        topk_logprobs: Number of top logprobs to request. Default: 20.
        verbose: Whether to show progress and print a summary. Default: True.

    Returns:
        List of lists of TokenMetrics, one per token in each output sequence.
    """
    # Check if any outputs have tokens to verify
    has_tokens = any(len(req.output_token_ids) > 0 for req in outputs)
    if not has_tokens:
        return [[] for _ in outputs]

    results = list(
        _iter_tinker_logprobs(
            outputs=outputs,
            sampling_client=sampling_client,
            topk_logprobs=topk_logprobs,
            verbose=verbose,
        )
    )

    return _process_results_to_metrics(
        results=results,
        n_outputs=len(outputs),
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        verbose=verbose,
    )


@torch.inference_mode()
async def verify_outputs_fireworks(
    outputs: list[TokenSequence],
    vocab_size: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    client: AsyncOpenAI,
    model: str,
    topk_logprobs: int = 20,
    verbose: bool = True,
    concurrency: int = 10,
) -> list[list[TokenMetrics]]:
    """
    Verify LLM outputs using Fireworks API (async).

    This function takes token sequences (prompt + generated output) and verifies
    whether the outputs could have been produced by the specified model using
    the given sampling parameters. Requests are submitted concurrently.

    In scripts, use: asyncio.run(verify_outputs_fireworks(...))
    In Jupyter/async contexts, use: await verify_outputs_fireworks(...)

    Args:
        outputs: List of TokenSequence objects containing prompt and output token IDs.
        vocab_size: The vocabulary size of the model.
        temperature: Sampling temperature used during generation.
        top_k: Top-k sampling parameter.
        top_p: Top-p (nucleus) sampling parameter.
        seed: Random seed used during generation.
        client: An AsyncOpenAI client configured for Fireworks.
        model: The Fireworks model name.
        topk_logprobs: Number of top logprobs to request. Default: 20.
        verbose: Whether to show progress and print a summary. Default: True.
        concurrency: Maximum number of concurrent requests. Default: 10.

    Returns:
        List of lists of TokenMetrics, one per token in each output sequence.
    """
    # Check if any outputs have tokens to verify
    has_tokens = any(len(req.output_token_ids) > 0 for req in outputs)
    if not has_tokens:
        return [[] for _ in outputs]

    results = await _fetch_all_fireworks_logprobs(
        outputs=outputs,
        client=client,
        model=model,
        topk_logprobs=topk_logprobs,
        verbose=verbose,
        concurrency=concurrency,
    )

    return _process_results_to_metrics(
        results=results,
        n_outputs=len(outputs),
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        verbose=verbose,
    )
