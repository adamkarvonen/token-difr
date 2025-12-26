"""End-to-end tests for token-difr using Tinker API backend."""

import os

import pytest
from transformers import AutoTokenizer

from token_difr import TokenSequence, compute_metrics_summary, verify_outputs_tinker

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_NAME = "Qwen/Qwen3-8B"
# MODEL_NAME = "openai/gpt-oss-120b"
# MODEL_NAME = "moonshotai/Kimi-K2-Thinking"
# MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
# MODEL_NAME = "deepseek-ai/DeepSeek-V3.1"

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


def get_tinker_api_key():
    """Get Tinker API key from environment."""
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        pytest.skip("TINKER_API_KEY environment variable not set")
    return api_key


def generate_outputs_tinker(sampling_client, prompts, temperature, top_k, top_p, seed, max_tokens=100):
    """Generate outputs using Tinker API for testing.

    Returns:
        Tuple of (outputs, vocab_size) where outputs is a list of TokenSequence
        and vocab_size is derived from the tokenizer.
    """
    import tinker

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Prepare all prompts and submit requests in parallel
    prompt_token_ids_list = []
    futures = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        prompt_token_ids_list.append(prompt_token_ids)

        # Submit request (returns a Future)
        prompt_input = tinker.ModelInput.from_ints(prompt_token_ids)
        params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
        )
        future = sampling_client.sample(
            prompt=prompt_input,
            sampling_params=params,
            num_samples=1,
        )
        futures.append(future)

    # Collect all results
    outputs = []
    for prompt_token_ids, future in zip(prompt_token_ids_list, futures):
        result = future.result()
        generated_tokens = result.sequences[0].tokens
        outputs.append(
            TokenSequence(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=list(generated_tokens),
            )
        )

    return outputs, vocab_size


@pytest.mark.parametrize("temperature", [0.0])  # Only greedy for now until Tinker sampling is understood
def test_verify_outputs_tinker(temperature):
    """Test Tinker verification achieves >= 98% exact match and all metrics/summary fields are valid."""
    import tinker

    api_key = get_tinker_api_key()

    top_k = 20  # Tinker limits topk logprobs to 20
    top_p = 0.95
    seed = 42
    max_tokens = 100
    min_match_rate = 0.98

    # Create Tinker client
    service_client = tinker.ServiceClient(api_key=api_key)
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    # Generate outputs
    outputs, vocab_size = generate_outputs_tinker(
        sampling_client=sampling_client,
        prompts=TEST_PROMPTS,
        temperature=1e-8,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    assert total_tokens > 0, "Should generate at least some tokens"

    # Verify outputs using Tinker backend
    topk_logprobs = 20
    results = verify_outputs_tinker(
        outputs,
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        sampling_client=sampling_client,
        topk_logprobs=topk_logprobs,
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
    model_name = MODEL_NAME.split("/")[-1]
    print(f"\n{model_name} (Tinker->Tinker): exact match rate = {summary['exact_match_rate']:.2%}")
    assert summary["exact_match_rate"] >= min_match_rate, (
        f"{model_name}: exact match rate {summary['exact_match_rate']:.2%} is below {min_match_rate:.0%} threshold"
    )
