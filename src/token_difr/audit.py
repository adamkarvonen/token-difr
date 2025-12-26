"""High-level audit interface for verifying LLM provider outputs."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from token_difr.common import compute_metrics_summary
from token_difr.api import verify_outputs_fireworks
from token_difr.model_registry import FIREWORKS_MODEL_REGISTRY, get_openrouter_name
from token_difr.openrouter_api import generate_openrouter_responses, tokenize_openrouter_responses


@dataclass
class AuditResult:
    """Result of auditing a provider's outputs."""

    exact_match_rate: float
    avg_prob: float
    avg_margin: float
    avg_logit_rank: float
    avg_gumbel_rank: float
    infinite_margin_rate: float
    total_tokens: int
    n_sequences: int

    def __repr__(self) -> str:
        return (
            f"AuditResult({self.exact_match_rate:.1%} match rate, "
            f"{self.total_tokens} tokens across {self.n_sequences} sequences)"
        )


async def _audit_provider_async(
    conversations: list[list[dict[str, str]]],
    model: str,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
    seed: int = 42,
    top_k: int = 50,
    top_p: float = 0.95,
    concurrency: int = 20,
) -> AuditResult:
    """Async implementation of audit_provider."""
    # Get API keys
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY")

    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    if not fireworks_api_key:
        raise ValueError("FIREWORKS_API_KEY environment variable not set")

    # Get Fireworks model name from registry
    if model not in FIREWORKS_MODEL_REGISTRY:
        raise ValueError(
            f"Model {model} not in FIREWORKS_MODEL_REGISTRY. "
            f"Use register_fireworks_model() to add it first."
        )
    fireworks_model = FIREWORKS_MODEL_REGISTRY[model]

    # Get OpenRouter model name (uses registry or falls back to lowercase)
    openrouter_model = get_openrouter_name(model)

    # Create clients
    openrouter_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    fireworks_client = AsyncOpenAI(
        api_key=fireworks_api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Generate responses via OpenRouter
    responses = await generate_openrouter_responses(
        client=openrouter_client,
        conversations=conversations,
        model=openrouter_model,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        concurrency=concurrency,
    )

    # Tokenize responses
    sequences = tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens)

    # Verify via Fireworks
    results = await verify_outputs_fireworks(
        sequences,
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        client=fireworks_client,
        model=fireworks_model,
        topk_logprobs=5,
    )

    # Compute summary
    summary = compute_metrics_summary(results)

    return AuditResult(
        exact_match_rate=summary["exact_match_rate"],
        avg_prob=summary["avg_prob"],
        avg_margin=summary["avg_margin"],
        avg_logit_rank=summary["avg_logit_rank"],
        avg_gumbel_rank=summary["avg_gumbel_rank"],
        infinite_margin_rate=summary["infinite_margin_rate"],
        total_tokens=summary["total_tokens"],
        n_sequences=len(sequences),
    )


def audit_provider(
    conversations: list[list[dict[str, str]]],
    model: str,
    provider: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 100,
    seed: int = 42,
    top_k: int = 50,
    top_p: float = 0.95,
    concurrency: int = 20,
) -> AuditResult:
    """Audit a provider by generating responses and verifying them against Fireworks.

    This function:
    1. Generates responses via OpenRouter using the specified provider
    2. Tokenizes the responses using the model's tokenizer
    3. Verifies the token sequences against Fireworks logprobs
    4. Returns an AuditResult with verification metrics

    Args:
        conversations: List of conversations, where each conversation is a list of
            message dicts with 'role' and 'content' keys. Use construct_prompts()
            to get a default dataset.
        model: HuggingFace model name (e.g., "meta-llama/Llama-3.3-70B-Instruct").
            Must be in FIREWORKS_MODEL_REGISTRY.
        provider: OpenRouter provider to use (e.g., "groq", "moonshotai").
            If None, OpenRouter will choose automatically.
        temperature: Sampling temperature. Use 0.0 for deterministic outputs.
        max_tokens: Maximum tokens to generate per response.
        seed: Random seed for reproducibility.
        top_k: Top-k sampling parameter for verification.
        top_p: Top-p (nucleus) sampling parameter for verification.
        concurrency: Number of concurrent API requests.

    Returns:
        AuditResult with verification metrics.

    Raises:
        ValueError: If API keys are not set or model is not in registry.

    Example:
        >>> from token_difr import construct_prompts, audit_provider
        >>> prompts = construct_prompts(n_prompts=50, model_name="meta-llama/Llama-3.3-70B-Instruct")
        >>> result = audit_provider(prompts, "meta-llama/Llama-3.3-70B-Instruct", provider="groq")
        >>> print(result)
        AuditResult(98.3% match rate, 4521 tokens across 50 sequences)
    """
    return asyncio.run(
        _audit_provider_async(
            conversations=conversations,
            model=model,
            provider=provider,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_k=top_k,
            top_p=top_p,
            concurrency=concurrency,
        )
    )
