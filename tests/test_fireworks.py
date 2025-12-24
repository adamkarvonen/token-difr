"""End-to-end tests for token-difr using Fireworks API backend."""

import asyncio
import os
from dataclasses import dataclass

import openai
import pytest
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from token_difr import TokenSequence, compute_metrics_summary, encode_thinking_response, verify_outputs_api

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Fireworks model name and corresponding HuggingFace model for tokenizer
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
HF_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

# OpenRouter configuration
OPENROUTER_LLAMA_MODEL = "meta-llama/llama-3.3-70b-instruct"
OPENROUTER_KIMI_MODEL = "moonshotai/kimi-k2-thinking"
HF_KIMI_MODEL = "moonshotai/Kimi-K2-Thinking"
FIREWORKS_KIMI_MODEL = "fireworks/kimi-k2-thinking"

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


def get_fireworks_api_key():
    """Get Fireworks API key from environment."""
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY environment variable not set")
    return api_key


def generate_outputs_fireworks(
    client: OpenAI,
    tokenizer,
    prompts: list[str],
    temperature: float,
    max_tokens: int = 100,
    model: str = FIREWORKS_MODEL,
    verbose: bool = True,
) -> tuple[list[TokenSequence], int]:
    """Generate outputs using Fireworks API for testing.

    Returns:
        Tuple of (outputs, vocab_size) where outputs is a list of TokenSequence
        and vocab_size is derived from the tokenizer.
    """
    vocab_size = len(tokenizer)

    outputs = []
    iterator = tqdm(prompts, desc="Generating via Fireworks") if verbose else prompts
    for prompt in iterator:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        # Generate using Fireworks completions API
        response = client.completions.create(
            model=model,
            prompt=rendered,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        generated_text = response.choices[0].text
        generated_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)

        outputs.append(
            TokenSequence(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=generated_token_ids,
            )
        )

    return outputs, vocab_size


@pytest.mark.parametrize("temperature", [0.0])
def test_verify_outputs_fireworks(temperature):
    """Test Fireworks verification achieves >= 98% exact match and all metrics/summary fields are valid."""
    api_key = get_fireworks_api_key()

    top_k = 5  # Fireworks default
    top_p = 0.95
    seed = 42
    max_tokens = 100
    min_match_rate = 0.98

    # Create Fireworks client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Generate outputs
    outputs, vocab_size = generate_outputs_fireworks(
        client=client,
        tokenizer=tokenizer,
        prompts=TEST_PROMPTS,
        temperature=1e-8,  # Near-greedy for generation
        max_tokens=max_tokens,
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    assert total_tokens > 0, "Should generate at least some tokens"

    # Verify outputs using Fireworks backend
    results = verify_outputs_api(
        outputs,
        backend="fireworks",
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        client=client,
        model=FIREWORKS_MODEL,
        tokenizer=tokenizer,
        topk_logprobs=5,
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
    print(f"\nTemperature {temperature}: exact match rate = {summary['exact_match_rate']:.2%}")
    assert summary["exact_match_rate"] >= min_match_rate, (
        f"Temperature {temperature}: exact match rate {summary['exact_match_rate']:.2%} "
        f"is below {min_match_rate:.0%} threshold"
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


async def openrouter_request(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    provider: str,
) -> tuple[str, str]:
    """Make a single OpenRouter request.

    Returns:
        Tuple of (content, reasoning) where reasoning may be empty for non-thinking models.
    """
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={
            "provider": {"order": [provider]},
        },
    )
    content = completion.choices[0].message.content or ""
    reasoning = getattr(completion.choices[0].message, "reasoning", None) or ""
    return content, reasoning


async def generate_outputs_openrouter(
    openrouter_client: openai.AsyncOpenAI,
    tokenizer,
    prompts: list[str],
    openrouter_model: str,
    provider: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
    concurrency: int = 8,
    verbose: bool = True,
) -> tuple[list[TokenSequence], int]:
    """Generate outputs using OpenRouter API.

    Args:
        openrouter_client: OpenRouter AsyncOpenAI client.
        tokenizer: HuggingFace tokenizer.
        prompts: List of prompts.
        openrouter_model: Model name on OpenRouter.
        provider: OpenRouter provider (e.g., "Fireworks").
        temperature: Sampling temperature.
        max_tokens: Max tokens to generate.
        concurrency: Max concurrent requests.
        verbose: Show progress bar.

    Returns:
        Tuple of (outputs, vocab_size).
    """
    vocab_size = len(tokenizer)
    semaphore = asyncio.Semaphore(concurrency)

    # Build conversations
    conversations = [[{"role": "user", "content": p}] for p in prompts]

    async def _wrapped(idx: int, messages: list[dict[str, str]]) -> tuple[int, str, str]:
        async with semaphore:
            content, reasoning = await openrouter_request(
                openrouter_client, openrouter_model, messages, max_tokens, temperature, provider
            )
            return idx, content, reasoning

    tasks = [asyncio.create_task(_wrapped(i, conv)) for i, conv in enumerate(conversations)]
    results: list[tuple[str, str]] = [("", "")] * len(conversations)

    if verbose:
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"OpenRouter ({provider})"):
            idx, content, reasoning = await fut
            results[idx] = (content, reasoning)
    else:
        for fut in asyncio.as_completed(tasks):
            idx, content, reasoning = await fut
            results[idx] = (content, reasoning)

    # Convert to TokenSequence
    outputs = []
    for conv, (content, reasoning) in zip(conversations, results):
        rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        response_token_ids = encode_thinking_response(content, reasoning, tokenizer, max_tokens)

        outputs.append(
            TokenSequence(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=response_token_ids,
            )
        )

    return outputs, vocab_size


@pytest.mark.asyncio
async def test_verify_openrouter_generation_with_fireworks():
    """Test: Generate via OpenRouter (Fireworks provider), verify via Fireworks API."""
    fireworks_api_key = get_fireworks_api_key()
    openrouter_api_key = get_openrouter_api_key()

    top_k = 5
    top_p = 0.95
    seed = 42
    max_tokens = 100
    min_match_rate = 0.98

    # Create clients
    fireworks_client = OpenAI(
        api_key=fireworks_api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )
    openrouter_client = openai.AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Generate via OpenRouter with Fireworks as provider
    outputs, vocab_size = await generate_outputs_openrouter(
        openrouter_client=openrouter_client,
        tokenizer=tokenizer,
        prompts=TEST_PROMPTS,
        openrouter_model=OPENROUTER_LLAMA_MODEL,
        provider="Fireworks",
        temperature=0.0,
        max_tokens=max_tokens,
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    assert total_tokens > 0, "Should generate at least some tokens"

    # Verify via Fireworks
    results = verify_outputs_api(
        outputs,
        backend="fireworks",
        vocab_size=vocab_size,
        temperature=0.0,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        client=fireworks_client,
        model=FIREWORKS_MODEL,
        tokenizer=tokenizer,
        topk_logprobs=5,
    )

    summary = compute_metrics_summary(results)
    print(f"\nOpenRouter->Fireworks: exact match rate = {summary['exact_match_rate']:.2%}")

    assert summary["exact_match_rate"] >= min_match_rate, (
        f"Exact match rate {summary['exact_match_rate']:.2%} is below {min_match_rate:.0%} threshold"
    )


# =============================================================================
# Echo Tests - Verify Fireworks doesn't modify formatted prompts
# =============================================================================


@dataclass
class EchoModelConfig:
    """Configuration for echo testing a model."""

    fireworks_model: str
    hf_model: str
    skip_first_token: bool  # Whether Fireworks prepends an empty token
    is_thinking_model: bool  # Whether the model uses <think>...</think> tags


# Model configurations for echo tests
ECHO_MODEL_CONFIGS = {
    "llama": EchoModelConfig(
        fireworks_model=FIREWORKS_MODEL,
        hf_model=HF_MODEL_NAME,
        skip_first_token=True,  # Llama has BOS token rendered as empty
        is_thinking_model=False,
    ),
    "kimi": EchoModelConfig(
        fireworks_model=FIREWORKS_KIMI_MODEL,
        hf_model=HF_KIMI_MODEL,
        skip_first_token=False,  # Kimi doesn't have this issue
        is_thinking_model=True,
    ),
}


def get_echo_test_messages(is_thinking_model: bool) -> list[dict[str, str]]:
    """Get test messages appropriate for the model type."""
    if is_thinking_model:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>Thinking...</think>The answer is 4."},
        ]
    else:
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "How are you?"},
        ]


@pytest.mark.parametrize("model_key", ["llama", "kimi"])
def test_echo(model_key: str):
    """Test that Fireworks echoes prompts without modification.

    Verifies:
    1. The number of echoed tokens matches expected (no hidden prefix/suffix)
    2. The non-special tokens match what we expect
    """
    config = ECHO_MODEL_CONFIGS[model_key]
    verbose = True

    api_key = get_fireworks_api_key()

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.hf_model, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load {config.hf_model} tokenizer: {e}")

    # Get appropriate test messages for this model type
    messages = get_echo_test_messages(config.is_thinking_model)
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    # Send with echo=True to get the prompt tokens back
    try:
        response = client.completions.create(
            model=config.fireworks_model,
            prompt=formatted_prompt,
            max_tokens=1,
            temperature=0.0,
            echo=True,
            logprobs=5,
        )
    except Exception as e:
        pytest.skip(f"{model_key} model not available on Fireworks: {e}")

    # The echoed tokens should include our prompt
    echoed_tokens = response.choices[0].logprobs.tokens

    # Some models (like Llama) have Fireworks prepend an empty token for BOS
    if config.skip_first_token:
        echoed_tokens = echoed_tokens[1:]

    # Exclude the generated token at the end
    prompt_tokens = echoed_tokens[: len(expected_tokens)]

    print(f"\n[{model_key}] Expected {len(expected_tokens)} tokens, got {len(prompt_tokens)} echoed tokens")

    # Key check: token count should match (no hidden prefix/suffix added)
    assert len(prompt_tokens) == len(expected_tokens), (
        f"[{model_key}] Token count mismatch: expected {len(expected_tokens)}, got {len(prompt_tokens)}. "
        "Fireworks may be adding hidden tokens to the prompt."
    )

    # Convert expected token IDs to strings for comparison
    expected_token_strings = [tokenizer.decode([tid]) for tid in expected_tokens]

    # Check for mismatches (special tokens may render differently, which is OK)
    mismatches = []
    special_token_mismatches = 0
    for i, (expected, echoed) in enumerate(zip(expected_token_strings, prompt_tokens)):
        if verbose:
            print(f"Expected: {expected}, Echoed: {echoed}")
        if expected != echoed:
            # Check if this is a special token rendering difference
            is_special = (echoed == "" and expected.startswith("<") and expected.endswith(">")) or (
                expected.startswith("<|") and expected.endswith("|>")
            )
            if is_special:
                special_token_mismatches += 1
            else:
                mismatches.append(f"  Position {i}: expected {repr(expected)}, got {repr(echoed)}")

    if mismatches:
        print(f"\nNon-special token mismatches ({len(mismatches)}):")
        for m in mismatches[:10]:
            print(m)

    if special_token_mismatches:
        print(f"Special token rendering differences: {special_token_mismatches} (expected)")

    # Only fail on non-special token mismatches
    assert len(mismatches) == 0, f"Found {len(mismatches)} unexpected token mismatches"
    print(f"\n[{model_key}] Echo test: PASSED (token counts match, no unexpected modifications)")
