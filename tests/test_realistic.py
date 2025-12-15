"""Test with more tokens to see realistic exact match rates."""

import gc

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from token_difr import TokenMetrics, TokenSequence, verify_outputs

import sys

# Allow model to be specified via command line
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"

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


def generate_outputs_vllm(model_name, prompts, temperature, top_k, top_p, seed, max_tokens=100):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
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


def compute_exact_match_rate(results):
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


if __name__ == "__main__":
    for temp in [0.0, 1.0]:
        print(f"\n{'='*60}")
        print(f"Temperature {temp}, max_tokens=500")
        print("=" * 60)

        outputs = generate_outputs_vllm(
            model_name=MODEL_NAME,
            prompts=TEST_PROMPTS,
            temperature=temp,
            top_k=50,
            top_p=0.95,
            seed=42,
            max_tokens=500,
        )

        total_tokens = sum(len(o.output_token_ids) for o in outputs)
        print(f"Generated {total_tokens} total tokens across {len(outputs)} sequences")

        results = verify_outputs(
            outputs,
            model_name=MODEL_NAME,
            temperature=temp,
            top_k=50,
            top_p=0.95,
            seed=42,
        )

        rate = compute_exact_match_rate(results)
        matches = int(rate * total_tokens)
        print(f"Exact match rate: {rate:.4%} ({matches}/{total_tokens})")

        if rate < 0.98:
            print(f"WARNING: Match rate {rate:.2%} is below 98% threshold!")
        else:
            print("PASSED: Match rate >= 98%")
