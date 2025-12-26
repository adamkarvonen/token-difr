"""Common utilities for token verification."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch


def encode_thinking_response(
    content: str,
    reasoning: str | None,
    tokenizer,
    max_tokens: int | None = None,
) -> list[int]:
    """Encode a thinking model response to token IDs.

    For thinking models (Kimi-K2, QwQ, DeepSeek-R1, etc.), the model outputs:
        <think>reasoning</think>content

    OpenRouter returns reasoning in a separate field, so we need to reconstruct
    the full token sequence.

    Args:
        content: The main response content.
        reasoning: The reasoning/thinking content (may be None or empty).
        tokenizer: HuggingFace tokenizer for the model.
        max_tokens: Optional maximum number of tokens to return.

    Returns:
        List of token IDs representing the full response.
    """
    if reasoning:
        full_response = f"<think>{reasoning}</think>{content}"
    else:
        full_response = content

    token_ids = tokenizer.encode(full_response, add_special_tokens=False)

    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]

    return token_ids


def construct_prompts(
    n_prompts: int = 100,
    max_ctx_len: int = 512,
    tokenizer=None,
    model_name: str | None = None,
    custom_prompts: list[list[dict[str, str]]] | None = None,
    system_prompt: str | None = "You are a helpful assistant.",
) -> list[list[dict[str, str]]]:
    """Construct a dataset of conversation prompts for testing.

    By default, loads prompts from the WildChat dataset (lmsys/lmsys-chat-1m),
    filtering for English conversations that end with a user message.

    Args:
        n_prompts: Number of prompts to return.
        max_ctx_len: Maximum context length in tokens (prompts longer than this are skipped).
        tokenizer: HuggingFace tokenizer for filtering by length. If None and model_name
            is provided, will load the tokenizer automatically.
        model_name: Model name to load tokenizer from if tokenizer is None.
        custom_prompts: Optional list of custom prompts to use instead of loading
            from the dataset. If provided, other parameters are ignored.
        system_prompt: System prompt to prepend to each conversation. Defaults to
            "You are a helpful assistant." Set to None to skip adding a system prompt.

    Returns:
        List of conversation prompts, where each prompt is a list of message dicts
        with 'role' and 'content' keys.
    """
    if custom_prompts is not None:
        prompts = custom_prompts[:n_prompts]
        if system_prompt is not None:
            prompts = [[{"role": "system", "content": system_prompt}] + p for p in prompts]
        return prompts

    # Import here to avoid dependency issues if not using this function
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Load or create tokenizer
    if tokenizer is None:
        if model_name is None:
            raise ValueError("Either tokenizer or model_name must be provided")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    ds = load_dataset("allenai/WildChat-1M", split="train")

    conversation_prompts: list[list[dict[str, str]]] = []
    unique_prompts: set[tuple[int, ...]] = set()

    idx = 0
    while len(conversation_prompts) < n_prompts and idx < len(ds):
        sample = ds[idx]
        idx += 1

        # Filter for English only
        if sample["language"].lower() != "english":  # type: ignore[index]
            continue

        # Get conversation and ensure it ends with a user message
        # Only keep 'role' and 'content' fields to avoid non-serializable types
        raw_conversation = sample["conversation"]  # type: ignore[index]
        conversation = [{"role": msg["role"], "content": msg["content"]} for msg in raw_conversation]

        while conversation and conversation[-1].get("role") == "assistant":
            conversation = conversation[:-1]

        if not conversation or conversation[-1].get("role") != "user":
            continue

        # Skip conversations with empty messages
        if any(not msg.get("content") or not msg["content"].strip() for msg in conversation):
            continue

        # Tokenize to check length and uniqueness
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        if len(token_ids) <= max_ctx_len:
            prompt_key = tuple(token_ids)
            if prompt_key not in unique_prompts:
                unique_prompts.add(prompt_key)
                conversation_prompts.append(conversation)

    if system_prompt is not None:
        conversation_prompts = [[{"role": "system", "content": system_prompt}] + c for c in conversation_prompts]

    return conversation_prompts


class SamplingMethod(Enum):
    """Supported sampling methods for verification."""

    VLLM_GUMBEL_MAX = "vllm_gumbel_max"
    TINKER_GUMBEL_MAX = "tinker_gumbel_max"


@dataclass
class TokenSequence:
    """A prompt and its generated output as token IDs."""

    prompt_token_ids: list[int]
    output_token_ids: list[int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TokenSequence":
        """Create from dictionary."""
        return cls(
            prompt_token_ids=d["prompt_token_ids"],
            output_token_ids=d["output_token_ids"],
        )


@dataclass
class TokenMetrics:
    """Verification metrics for a single token."""

    exact_match: bool
    prob: float
    margin: float
    logit_rank: float
    gumbel_rank: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


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


def compute_metrics_summary(results: list[list[TokenMetrics]], verbose: bool = False) -> dict[str, Any]:
    """
    Compute aggregate statistics from verification results.

    Args:
        results: Output from verify_outputs().
        verbose: If True, print a summary of the metrics.

    Returns:
        Dictionary with keys:
            - total_tokens: Total number of tokens verified
            - exact_match_rate: Fraction of tokens that matched exactly
            - avg_prob: Average probability of actual tokens
            - avg_margin: Average margin for non-infinite values
            - infinite_margin_rate: Fraction of tokens with infinite margin
            - avg_logit_rank: Average logit rank of actual tokens
            - avg_gumbel_rank: Average Gumbel rank of actual tokens
    """
    total_tokens = 0
    total_matches = 0
    total_prob = 0.0
    total_margin = 0.0
    finite_margin_count = 0
    total_logit_rank = 0.0
    total_gumbel_rank = 0.0

    for seq_metrics in results:
        for m in seq_metrics:
            total_tokens += 1
            if m.exact_match:
                total_matches += 1
            total_prob += m.prob
            if math.isfinite(m.margin):
                total_margin += m.margin
                finite_margin_count += 1
            total_logit_rank += m.logit_rank
            total_gumbel_rank += m.gumbel_rank

    if total_tokens == 0:
        return {
            "total_tokens": 0,
            "exact_match_rate": 0.0,
            "avg_prob": 0.0,
            "avg_margin": 0.0,
            "infinite_margin_rate": 0.0,
            "avg_logit_rank": 0.0,
            "avg_gumbel_rank": 0.0,
        }

    infinite_margin_count = total_tokens - finite_margin_count
    avg_margin = total_margin / finite_margin_count if finite_margin_count > 0 else float("inf")

    summary = {
        "total_tokens": total_tokens,
        "exact_match_rate": total_matches / total_tokens,
        "avg_prob": total_prob / total_tokens,
        "avg_margin": avg_margin,
        "infinite_margin_rate": infinite_margin_count / total_tokens,
        "avg_logit_rank": total_logit_rank / total_tokens,
        "avg_gumbel_rank": total_gumbel_rank / total_tokens,
    }

    if verbose:
        print("Verification Summary:")
        print(f"  Total tokens: {summary['total_tokens']}")
        print(f"  Exact match rate: {summary['exact_match_rate']:.2%}")
        print(f"  Average probability: {summary['avg_prob']:.4f}")
        print(f"  Average margin: {summary['avg_margin']:.4f} ({summary['infinite_margin_rate']:.2%} infinite)")

    return summary
