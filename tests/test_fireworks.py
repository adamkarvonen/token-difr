"""End-to-end tests for token-difr using Fireworks API backend."""

import asyncio
import os
from dataclasses import dataclass

import openai
import pytest
from openai import AsyncOpenAI, OpenAI
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
from transformers import AutoTokenizer

from token_difr import TokenSequence, compute_metrics_summary, encode_thinking_response, verify_outputs_fireworks
from token_difr.openrouter_api import openrouter_request

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Model configurations keyed by HuggingFace model name
# Each entry: {hf_model: (fireworks_model, openrouter_model)}
MODEL_CONFIGS = {
    "meta-llama/Llama-3.3-70B-Instruct": (
        "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
    ),
    "moonshotai/Kimi-K2-Thinking": (
        "accounts/fireworks/models/kimi-k2-thinking",
        "moonshotai/kimi-k2-thinking",
    ),
}

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a haiku about the ocean.",
    "What is 2 + 2?",
    "List three primary colors.",
    "Describe the water cycle.",
    "What causes rainbows?",
    "Explain gravity to a child.",
]

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def get_fireworks_api_key():
    """Get Fireworks API key from environment."""
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY environment variable not set")
    return api_key


async def generate_outputs_fireworks(
    client: AsyncOpenAI,
    tokenizer,
    prompts: list[str],
    temperature: float,
    model: str,
    max_tokens: int = 100,
    verbose: bool = True,
    concurrency: int = 10,
) -> tuple[list[TokenSequence], int]:
    """Generate outputs using Fireworks API for testing.

    Returns:
        Tuple of (outputs, vocab_size) where outputs is a list of TokenSequence
        and vocab_size is derived from the tokenizer.
    """
    vocab_size = len(tokenizer)
    semaphore = asyncio.Semaphore(concurrency)

    async def generate_one(prompt: str) -> TokenSequence:
        async with semaphore:
            messages = [{"role": "user", "content": prompt}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)

            # Generate using Fireworks completions API with token IDs
            # Use echo=True and logprobs to get actual token IDs back
            response = await client.completions.create(
                model=model,
                prompt=prompt_token_ids,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=True,
                logprobs=True,
                extra_body={"echo": True, "top_logprobs": 1},
            )

            generated_tokens = response.choices[0].logprobs.content[len(prompt_token_ids) :]
            generated_token_ids = [content["token_id"] for content in generated_tokens]

            return TokenSequence(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=generated_token_ids,
            )

    # Run all generations concurrently
    tasks = [generate_one(prompt) for prompt in prompts]

    if verbose:
        outputs = []
        for coro in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating via Fireworks"):
            outputs.append(await coro)
    else:
        outputs = await asyncio.gather(*tasks)

    return outputs, vocab_size


@pytest.mark.parametrize(
    "hf_model",
    list(MODEL_CONFIGS.keys()),
    ids=[k.split("/")[-1] for k in MODEL_CONFIGS.keys()],
)
def test_verify_outputs_fireworks(hf_model):
    """Test Fireworks verification achieves >= 98% exact match and all metrics/summary fields are valid."""
    api_key = get_fireworks_api_key()
    fireworks_model, _ = MODEL_CONFIGS[hf_model]
    model_name = hf_model.split("/")[-1]

    top_k = 5  # Fireworks default
    top_p = 0.95
    seed = 42
    max_tokens = 100
    temperature = 0.0
    min_match_rate = 0.97

    # Create Fireworks async client (for generation)
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Generate outputs (async, run with asyncio.run)
    outputs, vocab_size = asyncio.run(
        generate_outputs_fireworks(
            client=client,
            tokenizer=tokenizer,
            prompts=TEST_PROMPTS,
            temperature=0,
            max_tokens=max_tokens,
            model=fireworks_model,
        )
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    assert total_tokens > 0, "Should generate at least some tokens"

    # Verify outputs using Fireworks backend (async)
    results = asyncio.run(
        verify_outputs_fireworks(
            outputs,
            vocab_size=vocab_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            client=client,
            model=fireworks_model,
            topk_logprobs=5,
        )
    )

    # Check results structure
    assert len(results) == len(outputs), "Should have results for each output sequence"
    for seq_idx, seq_results in enumerate(results):
        assert len(seq_results) == len(outputs[seq_idx].output_token_ids), (
            f"Sequence {seq_idx}: should have metrics for each token"
        )

    # Check TokenMetrics fields
    for seq_results in results:
        for metrics in seq_results:
            assert isinstance(metrics.exact_match, bool)
            assert isinstance(metrics.prob, float)
            assert isinstance(metrics.margin, float)
            assert isinstance(metrics.logit_rank, (int, float))
            assert isinstance(metrics.gumbel_rank, (int, float))
            assert 0.0 <= metrics.prob <= 1.0, f"prob should be in [0, 1], got {metrics.prob}"
            assert metrics.logit_rank >= 0, f"logit_rank should be >= 0, got {metrics.logit_rank}"
            assert metrics.gumbel_rank >= 0, f"gumbel_rank should be >= 0, got {metrics.gumbel_rank}"

    # Check compute_metrics_summary
    summary = compute_metrics_summary(results)
    expected_keys = [
        "total_tokens",
        "exact_match_rate",
        "avg_prob",
        "avg_margin",
        "infinite_margin_rate",
        "avg_logit_rank",
        "avg_gumbel_rank",
    ]
    for key in expected_keys:
        assert key in summary, f"Missing key: {key}"
    assert summary["total_tokens"] == total_tokens
    assert 0.0 <= summary["exact_match_rate"] <= 1.0
    assert 0.0 <= summary["avg_prob"] <= 1.0
    assert 0.0 <= summary["infinite_margin_rate"] <= 1.0
    assert summary["avg_logit_rank"] >= 0.0
    assert summary["avg_gumbel_rank"] >= 0.0

    # Check match rate
    print(f"\n{model_name} (Fireworks->Fireworks): exact match rate = {summary['exact_match_rate']:.2%}")
    assert summary["exact_match_rate"] >= min_match_rate, (
        f"{model_name}: exact match rate {summary['exact_match_rate']:.2%} is below {min_match_rate:.0%} threshold"
    )


# =============================================================================
# OpenRouter Generation Tests
# =============================================================================


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    return api_key


async def generate_outputs_openrouter(
    client: openai.AsyncOpenAI,
    tokenizer,
    prompts: list[str],
    model: str,
    provider: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
    concurrency: int = 8,
    seed: int | None = None,
) -> tuple[list[TokenSequence], int]:
    """Generate outputs using OpenRouter API.

    Uses openrouter_request from token_difr.openrouter_api.

    Returns:
        Tuple of (outputs, vocab_size).
    """
    vocab_size = len(tokenizer)
    semaphore = asyncio.Semaphore(concurrency)

    # Add a default system prompt to ensure consistent behavior between OpenRouter and
    # HuggingFace tokenization. Without an explicit system prompt, HuggingFace's
    # apply_chat_template may add a model-specific default, while OpenRouter won't.
    default_system_prompt = "You are a helpful assistant."
    conversations = [
        [{"role": "system", "content": default_system_prompt}, {"role": "user", "content": p}] for p in prompts
    ]

    async def _wrapped(idx: int, messages: list[dict[str, str]]) -> tuple[int, str, str]:
        async with semaphore:
            content, reasoning = await openrouter_request(
                client, model, messages, max_tokens, temperature, provider, seed=seed
            )
            return idx, content, reasoning

    tasks = [asyncio.create_task(_wrapped(i, conv)) for i, conv in enumerate(conversations)]
    results: list[tuple[str, str]] = [("", "")] * len(conversations)

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"OpenRouter ({provider})"):
        idx, content, reasoning = await fut
        results[idx] = (content, reasoning)

    # Convert to TokenSequence
    outputs = []
    for conv, (content, reasoning) in zip(conversations, results):
        rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        response_token_ids = encode_thinking_response(content, reasoning, tokenizer, max_tokens)
        outputs.append(TokenSequence(prompt_token_ids=prompt_token_ids, output_token_ids=response_token_ids))

    return outputs, vocab_size


@pytest.mark.parametrize(
    "hf_model",
    list(MODEL_CONFIGS.keys()),
    ids=[k.split("/")[-1] for k in MODEL_CONFIGS.keys()],
)
def test_verify_openrouter_generation_with_fireworks(hf_model):
    """Test: Generate via OpenRouter, verify via Fireworks API."""
    fireworks_api_key = get_fireworks_api_key()
    openrouter_api_key = get_openrouter_api_key()
    fireworks_model, openrouter_model = MODEL_CONFIGS[hf_model]
    model_name = hf_model.split("/")[-1]

    temperature = 0.0
    top_k = 50
    topk_logprobs = 5
    top_p = 0.95
    seed = 42
    max_tokens = 100
    min_match_rate = 0.95

    # Create clients
    fireworks_client = AsyncOpenAI(
        api_key=fireworks_api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )
    openrouter_client = openai.AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Generate via OpenRouter with Fireworks as provider
    outputs, vocab_size = asyncio.run(
        generate_outputs_openrouter(
            client=openrouter_client,
            tokenizer=tokenizer,
            prompts=TEST_PROMPTS,
            model=openrouter_model,
            provider="fireworks",
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
        )
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    assert total_tokens > 0, "Should generate at least some tokens"

    # Verify via Fireworks (async)
    results = asyncio.run(
        verify_outputs_fireworks(
            outputs,
            vocab_size=vocab_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            client=fireworks_client,
            model=fireworks_model,
            topk_logprobs=topk_logprobs,
        )
    )

    summary = compute_metrics_summary(results)
    print(f"\n{model_name} (OpenRouter->Fireworks): exact match rate = {summary['exact_match_rate']:.2%}")

    assert summary["exact_match_rate"] >= min_match_rate, (
        f"Exact match rate {summary['exact_match_rate']:.2%} is below {min_match_rate:.0%} threshold"
    )
