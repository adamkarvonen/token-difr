"""API-based verification using Tinker."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tqdm import tqdm

from token_difr.common import (
    TokenMetrics,
    TokenSequence,
    _as_list,
    compute_metrics_summary,
)

if TYPE_CHECKING:
    import tinker


def _tinker_logprobs_to_tensor(
    topk_prompt_logprobs: list[list[tuple[int, float]] | None],
    start_idx: int,
    n_tokens: int,
    device: torch.device,
    vocab_size: int,
) -> torch.Tensor:
    """Convert Tinker's topk_prompt_logprobs into a dense full-vocabulary tensor.

    Tinker returns logprobs as list[list[tuple[int, float]] | None] where each
    entry is a list of (token_id, logprob) tuples sorted by descending probability.
    """
    slice_rows = topk_prompt_logprobs[start_idx : start_idx + n_tokens]
    if len(slice_rows) != n_tokens:
        raise ValueError(f"Expected {n_tokens} prompt logprob rows, got {len(slice_rows)}")

    logits = torch.full((n_tokens, vocab_size), float("-inf"), device=device)

    for j, row in enumerate(slice_rows):
        if row is None:
            continue

        token_ids = torch.tensor([tok_id for tok_id, _ in row], device=device, dtype=torch.long)
        logprobs = torch.tensor([logprob for _, logprob in row], device=device)
        logits[j].scatter_(0, token_ids, logprobs)

    return logits


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

    This is used for Tinker backend which returns log probs rather than raw logits.
    Currently only supports greedy verification (temperature=0). The signature
    includes all sampling parameters for future expansion when Tinker's sampling
    method is understood.

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
def verify_outputs_tinker(
    outputs: list[TokenSequence],
    sampling_client: "tinker.SamplingClient",
    vocab_size: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    topk_logprobs: int = 20,
    verbose: bool = True,
) -> list[list[TokenMetrics]]:
    """
    Verify LLM outputs using Gumbel-Max sampling verification via Tinker API.

    This function takes token sequences (prompt + generated output) and verifies
    whether the outputs could have been produced by the specified model using
    the given sampling parameters. Uses the Tinker API to fetch logprobs.

    Args:
        outputs: List of TokenSequence objects containing prompt and output token IDs.
        sampling_client: A Tinker SamplingClient instance configured with the model.
        vocab_size: The vocabulary size of the model (e.g., 128256 for Llama 3.1).
        temperature: Sampling temperature used during generation. Required.
        top_k: Top-k sampling parameter. Required.
        top_p: Top-p (nucleus) sampling parameter. Required.
        seed: Random seed used during generation. Required.
        topk_logprobs: Number of top logprobs to request from Tinker. Default: 50.
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
        >>> import tinker
        >>> from token_difr import verify_outputs_tinker, TokenSequence
        >>> service_client = tinker.ServiceClient(api_key="your-api-key")
        >>> sampling_client = service_client.create_sampling_client(
        ...     base_model="meta-llama/Llama-3.1-8B-Instruct"
        ... )
        >>> outputs = [
        ...     TokenSequence(
        ...         prompt_token_ids=[128000, 2323, 374],
        ...         output_token_ids=[264, 1296, 13]
        ...     )
        ... ]
        >>> results = verify_outputs_tinker(
        ...     outputs,
        ...     sampling_client,
        ...     vocab_size=128256,
        ...     temperature=1.0,
        ...     top_k=50,
        ...     top_p=0.95,
        ...     seed=42,
        ... )
    """
    import tinker

    device = torch.device("cpu")  # Tinker does computation remotely; we only need CPU for metrics

    all_token_metrics: list[list[TokenMetrics]] = [[] for _ in outputs]

    iterator = enumerate(outputs)
    if verbose:
        iterator = enumerate(tqdm(outputs, desc="Verifying via Tinker API"))

    for i, req in iterator:
        prompt_token_ids: list[int] = _as_list(req.prompt_token_ids)
        gen_ids: list[int] = _as_list(req.output_token_ids)

        if len(gen_ids) == 0:
            continue

        # Concatenate prompt + generated tokens
        full_sequence = prompt_token_ids + gen_ids
        full_prompt = tinker.ModelInput.from_ints(full_sequence)

        # Get logprobs for the full sequence
        logprob_result = sampling_client.sample(
            prompt=full_prompt,
            sampling_params=tinker.SamplingParams(max_tokens=1),
            num_samples=1,
            include_prompt_logprobs=True,
            topk_prompt_logprobs=topk_logprobs,
        ).result()

        # Convert Tinker's logprob format to tensor
        prompt_len = len(prompt_token_ids)
        gen_len = len(gen_ids)

        logits_JV = _tinker_logprobs_to_tensor(
            logprob_result.topk_prompt_logprobs,
            start_idx=prompt_len,
            n_tokens=gen_len,
            device=device,
            vocab_size=vocab_size,
        )

        # Compute verification metrics using log probs (not raw logits)
        seq_token_metrics = _compute_verification_metrics_from_logprobs(
            logprobs_JV=logits_JV,  # These are actually log probs from Tinker
            gen_ids=gen_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            device=device,
        )

        all_token_metrics[i] = seq_token_metrics

    if verbose:
        summary = compute_metrics_summary(all_token_metrics)
        print("Verification Summary:")
        print(f"  Total tokens: {summary['total_tokens']}")
        print(f"  Exact match rate: {summary['exact_match_rate']:.2%}")
        print(f"  Average probability: {summary['avg_prob']:.4f}")
        print(f"  Average margin: {summary['avg_margin']:.4f} ({summary['infinite_margin_rate']:.2%} infinite)")

    return all_token_metrics
