"""End-to-end tests for token-difr."""

import gc
import os

import pytest
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from token_difr import TokenSequence, compute_metrics_summary, verify_outputs

MODEL_NAME = os.environ.get("TEST_MODEL", "Qwen/Qwen3-1.7B")

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


def generate_outputs_vllm(prompts, temperature, top_k, top_p, seed, max_tokens=100):
    """Generate outputs using vLLM for testing."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        tokenized_prompts.append(token_ids)
    del tokenizer

    model = LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        max_model_len=4096,
        enforce_eager=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.5,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    token_prompts = [{"prompt_token_ids": ids} for ids in tokenized_prompts]
    vllm_outputs = model.generate(token_prompts, sampling_params=sampling_params)

    outputs = [
        TokenSequence(
            prompt_token_ids=list(req.prompt_token_ids),
            output_token_ids=list(req.outputs[0].token_ids),
        )
        for req in vllm_outputs
    ]

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return outputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_verify_outputs(temperature):
    """Test verification achieves >= 98% exact match and all metrics/summary fields are valid."""
    top_k = 50
    top_p = 0.95
    seed = 42
    max_tokens = 500
    min_match_rate = 0.98

    # Generate outputs
    outputs = generate_outputs_vllm(
        prompts=TEST_PROMPTS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        max_tokens=max_tokens,
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    assert total_tokens > 0, "Should generate at least some tokens"

    # Verify outputs
    results = verify_outputs(
        outputs,
        model_name=MODEL_NAME,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
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
    assert summary["exact_match_rate"] >= min_match_rate, (
        f"Temperature {temperature}: exact match rate {summary['exact_match_rate']:.2%} "
        f"is below {min_match_rate:.0%} threshold"
    )
