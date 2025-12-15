"""End-to-end tests for token-difr verification."""

from __future__ import annotations

import gc

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from token_difr import TokenMetrics, TokenSequence, verify_outputs

MODEL_NAME = "Qwen/Qwen3-1.7B"
MAX_TOKENS = 500  # Generate enough tokens for realistic match rates

# Simple test prompts
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


def generate_outputs_vllm(
    model_name: str,
    prompts: list[str],
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    max_tokens: int = MAX_TOKENS,
) -> list[TokenSequence]:
    """Generate outputs using vLLM with Gumbel-Max sampling."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize prompts with chat template
    tokenized_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        tokenized_prompts.append(token_ids)

    del tokenizer

    model = LLM(
        model=model_name,
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
        seed=seed,  # Enables deterministic Gumbel-Max sampling
    )

    token_prompts: list[TokensPrompt] = [{"prompt_token_ids": ids} for ids in tokenized_prompts]
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


def compute_exact_match_rate(results: list[list[TokenMetrics]]) -> float:
    """Compute overall exact match rate across all tokens."""
    total_tokens = 0
    total_matches = 0

    for seq_metrics in results:
        for token_metrics in seq_metrics:
            total_tokens += 1
            if token_metrics.exact_match:
                total_matches += 1

    if total_tokens == 0:
        return 0.0

    return total_matches / total_tokens


def test_temperature_0():
    """Test verification with temperature 0 (greedy sampling)."""
    print("\n" + "=" * 60)
    print("Testing temperature=0 (greedy sampling)")
    print("=" * 60)

    temperature = 0.0
    top_k = 50
    top_p = 0.95
    seed = 42

    # Generate outputs
    print(f"Generating outputs with {MODEL_NAME}...")
    outputs = generate_outputs_vllm(
        model_name=MODEL_NAME,
        prompts=TEST_PROMPTS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    print(f"Generated {total_tokens} tokens across {len(outputs)} sequences")

    # Verify outputs
    print("\nVerifying outputs...")
    results = verify_outputs(
        outputs,
        model_name=MODEL_NAME,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    # Compute metrics
    exact_match_rate = compute_exact_match_rate(results)
    print(f"\nExact match rate: {exact_match_rate:.2%}")

    assert exact_match_rate >= 0.98, f"Expected >= 98% exact match rate, got {exact_match_rate:.2%}"
    print("PASSED: Temperature 0 test")

    return exact_match_rate


def test_temperature_1():
    """Test verification with temperature 1."""
    print("\n" + "=" * 60)
    print("Testing temperature=1")
    print("=" * 60)

    temperature = 1.0
    top_k = 50
    top_p = 0.95
    seed = 42

    # Generate outputs
    print(f"Generating outputs with {MODEL_NAME}...")
    outputs = generate_outputs_vllm(
        model_name=MODEL_NAME,
        prompts=TEST_PROMPTS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    total_tokens = sum(len(o.output_token_ids) for o in outputs)
    print(f"Generated {total_tokens} tokens across {len(outputs)} sequences")

    # Verify outputs
    print("\nVerifying outputs...")
    results = verify_outputs(
        outputs,
        model_name=MODEL_NAME,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    # Compute metrics
    exact_match_rate = compute_exact_match_rate(results)
    print(f"\nExact match rate: {exact_match_rate:.2%}")

    assert exact_match_rate >= 0.98, f"Expected >= 98% exact match rate, got {exact_match_rate:.2%}"
    print("PASSED: Temperature 1 test")

    return exact_match_rate


if __name__ == "__main__":
    print("=" * 60)
    print("Token-DIFR End-to-End Test")
    print(f"Model: {MODEL_NAME}")
    print(f"Max tokens per sequence: {MAX_TOKENS}")
    print("=" * 60)

    rate_t0 = test_temperature_0()
    rate_t1 = test_temperature_1()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Temperature 0 exact match rate: {rate_t0:.2%}")
    print(f"Temperature 1 exact match rate: {rate_t1:.2%}")
    print("\nAll tests passed!")
