"""Tests for the high-level audit_provider API."""

import os

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from token_difr import audit_provider, construct_prompts

# Model configurations: HuggingFace model name -> OpenRouter provider
MODEL_TO_PROVIDER = {
    # "meta-llama/Llama-3.3-70B-Instruct": "groq",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "parasail/fp8",
}

# Test configuration
N_PROMPTS = 100
MAX_TOKENS = 200
MIN_MATCH_RATE = 0.90  # Lower threshold since cross-provider verification may have variance

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def skip_if_missing_api_keys():
    """Skip test if required API keys are not set."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    if not os.environ.get("FIREWORKS_API_KEY"):
        pytest.skip("FIREWORKS_API_KEY environment variable not set")


@pytest.mark.parametrize(
    "hf_model",
    list(MODEL_TO_PROVIDER.keys()),
    ids=[k.split("/")[-1] for k in MODEL_TO_PROVIDER.keys()],
)
def test_audit_provider(hf_model):
    """Test audit_provider returns valid results and reasonable match rates."""
    skip_if_missing_api_keys()

    provider = MODEL_TO_PROVIDER[hf_model]
    model_name = hf_model.split("/")[-1]

    # Load prompts
    prompts = construct_prompts(
        n_prompts=N_PROMPTS,
        model_name=hf_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    assert len(prompts) == N_PROMPTS

    # Run audit
    result = audit_provider(
        prompts,
        model=hf_model,
        provider=provider,
        max_tokens=MAX_TOKENS,
    )

    # Check result structure
    assert result.total_tokens > 0
    assert result.n_sequences == N_PROMPTS
    assert 0.0 <= result.exact_match_rate <= 1.0
    assert 0.0 <= result.avg_prob <= 1.0
    assert result.avg_margin >= 0.0
    assert result.avg_logit_rank >= 0.0
    assert result.avg_gumbel_rank >= 0.0
    assert 0.0 <= result.infinite_margin_rate <= 1.0
    assert result.threshold == 0.97  # Default threshold

    # Check match rate is reasonable
    print(f"\n{model_name} ({provider}): exact match rate = {result.exact_match_rate:.2%}")
    assert result.exact_match_rate >= MIN_MATCH_RATE, (
        f"{model_name}: exact match rate {result.exact_match_rate:.2%} is below {MIN_MATCH_RATE:.0%} threshold"
    )
