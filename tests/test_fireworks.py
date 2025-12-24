"""End-to-end tests for token-difr using Fireworks API backend."""

import os

import pytest
from openai import OpenAI
from transformers import AutoTokenizer

from token_difr import TokenSequence, compute_metrics_summary, verify_outputs_api

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Fireworks model name and corresponding HuggingFace model for tokenizer
FIREWORKS_MODEL = "accounts/fireworks/models/llama-v3p3-70b-instruct"
HF_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"

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


def generate_outputs_fireworks(client, tokenizer, prompts, temperature, max_tokens=100):
    """Generate outputs using Fireworks API for testing.

    Returns:
        Tuple of (outputs, vocab_size) where outputs is a list of TokenSequence
        and vocab_size is derived from the tokenizer.
    """
    vocab_size = len(tokenizer)

    outputs = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        # Generate using Fireworks completions API
        response = client.completions.create(
            model=FIREWORKS_MODEL,
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
