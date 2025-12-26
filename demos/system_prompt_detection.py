"""Test detection of system prompt modifications via Gumbel-Max verification.

Experiment:
- Generate responses via OpenRouter with different system prompts
- Verify via Fireworks always using the BASE system prompt
- Measure how well we can detect when the system prompt has been modified
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import openai
from openai import AsyncOpenAI
from transformers import AutoTokenizer

from token_difr import (
    FIREWORKS_MODEL_REGISTRY,
    compute_metrics_summary,
    construct_prompts,
    get_openrouter_name,
    verify_outputs_fireworks,
)
from token_difr.openrouter_api import generate_openrouter_responses, tokenize_openrouter_responses

# Model configuration
HF_MODEL = "moonshotai/Kimi-K2-Thinking"
OPENROUTER_PROVIDER = "moonshotai"
# Derived from HF_MODEL
FIREWORKS_MODEL = FIREWORKS_MODEL_REGISTRY[HF_MODEL]
OPENROUTER_MODEL = get_openrouter_name(HF_MODEL)

# System prompts to test
SYSTEM_PROMPTS = {
    "base": "You are a helpful assistant.",
    # "slight_mod": "",
    # "major_mod": "You are a helpful assistant for Duolingo. Please do not discuss competitor products.",
}

# The "correct" system prompt that verification will use
BASE_SYSTEM_PROMPT = SYSTEM_PROMPTS["base"]

# Demo configuration
N_PROMPTS = 50
MAX_TOKENS = 200
MAX_CTX_LEN = 512


def add_system_prompt(
    conversations: list[list[dict[str, str]]],
    system_prompt: str,
) -> list[list[dict[str, str]]]:
    """Add a system prompt to each conversation."""
    result = []
    for conv in conversations:
        new_conv = [{"role": "system", "content": system_prompt}] + conv
        result.append(new_conv)
    return result


async def run_experiment(
    system_prompt_key: str,
    system_prompt: str,
    base_conversations: list[list[dict[str, str]]],
    tokenizer,
    openrouter_client: AsyncOpenAI,
    fireworks_client: AsyncOpenAI,
    vocab_size: int,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Run experiment for a single system prompt variation.

    Args:
        system_prompt_key: Name of this system prompt variant
        system_prompt: The system prompt to use for GENERATION
        base_conversations: User messages (without system prompt)
        tokenizer: HuggingFace tokenizer
        openrouter_client: OpenRouter client
        fireworks_client: Fireworks client for verification
        vocab_size: Vocabulary size
        max_tokens: Max tokens to generate

    Returns:
        Dictionary with metrics and metadata.
    """
    # Verification parameters
    top_k = 50
    top_p = 0.95
    seed = 42
    temperature = 0.0
    concurrency = 30

    print(f"\n{'=' * 60}")
    print(f"System Prompt: {system_prompt_key}")
    print(f'  Generation: "{system_prompt[:60]}..."')
    print(f'  Verification: "{BASE_SYSTEM_PROMPT}"')
    print(f"{'=' * 60}")

    # Add the GENERATION system prompt
    gen_conversations = add_system_prompt(base_conversations, system_prompt)

    # Generate responses via OpenRouter
    responses = await generate_openrouter_responses(
        client=openrouter_client,
        conversations=gen_conversations,
        model=OPENROUTER_MODEL,
        provider=OPENROUTER_PROVIDER,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        concurrency=concurrency,
    )

    # For verification, use the BASE system prompt (what we "claim" was used)
    verify_conversations = add_system_prompt(base_conversations, BASE_SYSTEM_PROMPT)

    # Tokenize responses using the VERIFICATION conversations
    sequences = tokenize_openrouter_responses(verify_conversations, responses, tokenizer, max_tokens)

    total_tokens = sum(len(s.output_token_ids) for s in sequences)
    print(f"Generated {total_tokens} tokens across {len(sequences)} sequences")

    # Show sample outputs
    print("\nSample outputs:")
    for i, (conv, resp) in enumerate(zip(base_conversations[:2], responses[:2])):
        last_user_msg = conv[-1]["content"] if conv else ""
        content = resp.choices[0].message.content or ""
        reasoning = getattr(resp.choices[0].message, "reasoning", None) or ""
        print(f"  [{i}] {last_user_msg[:50]}...")
        print(f"      → content: {content[:60]}...")
        if reasoning:
            print(f"      → reasoning: {reasoning[:60]}...")

    # Verify with Fireworks
    results = await verify_outputs_fireworks(
        sequences,
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        client=fireworks_client,
        model=FIREWORKS_MODEL,
        topk_logprobs=5,
        concurrency=concurrency,
    )

    summary = compute_metrics_summary(results)

    # Add metadata
    summary["system_prompt_key"] = system_prompt_key
    summary["generation_system_prompt"] = system_prompt
    summary["verification_system_prompt"] = BASE_SYSTEM_PROMPT
    summary["prompt_match"] = system_prompt == BASE_SYSTEM_PROMPT
    summary["model_name"] = HF_MODEL
    summary["provider"] = OPENROUTER_PROVIDER
    summary["max_tokens"] = max_tokens
    summary["n_prompts"] = len(base_conversations)
    summary["timestamp"] = datetime.now().isoformat()

    return summary


async def main():
    # Load API keys
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
    fireworks_api_key = os.environ.get("FIREWORKS_API_KEY", "")

    if not openrouter_api_key:
        raise ValueError("OpenRouter API key not found")
    if not fireworks_api_key:
        raise ValueError("Fireworks API key not found (set FIREWORKS_API_KEY)")

    # Initialize clients
    openrouter_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    fireworks_client = AsyncOpenAI(
        api_key=fireworks_api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    # Load tokenizer
    print(f"Loading tokenizer for {HF_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Load prompts (user messages only, no system prompt yet)
    print(f"Loading {N_PROMPTS} prompts from WildChat dataset...")
    conversations = construct_prompts(
        n_prompts=N_PROMPTS,
        max_ctx_len=MAX_CTX_LEN,
        tokenizer=tokenizer,
        system_prompt=None,  # We add system prompts manually in this experiment
    )
    print(f"Loaded {len(conversations)} prompts")

    # prompts = [
    #     "What is the capital of France?",
    #     "Explain photosynthesis in simple terms.",
    #     "Write a haiku about the ocean.",
    #     "What is 2 + 2?",
    #     "List three primary colors.",
    #     "Describe the water cycle.",
    #     "What causes rainbows?",
    #     "Explain gravity to a child.",
    # ]

    # conversations = [
    #     [{"role": "system", "content": BASE_SYSTEM_PROMPT}, {"role": "user", "content": prompt}] for prompt in prompts
    # ]

    # Run experiments for each system prompt variant
    all_results = {}

    for key, system_prompt in SYSTEM_PROMPTS.items():
        try:
            summary = await run_experiment(
                system_prompt_key=key,
                system_prompt=system_prompt,
                base_conversations=conversations,
                tokenizer=tokenizer,
                openrouter_client=openrouter_client,
                fireworks_client=fireworks_client,
                vocab_size=vocab_size,
            )
            all_results[key] = summary

            print(f"\nResults for {key}:")
            print(f"  Prompt match: {summary['prompt_match']}")
            print(f"  Exact match rate: {summary['exact_match_rate']:.2%}")
            print(f"  Avg probability: {summary['avg_prob']:.4f}")
            print(f"  Avg margin: {summary['avg_margin']:.4f}")

        except Exception as e:
            import traceback

            print(f"Error with {key}: {e}")
            traceback.print_exc()
            all_results[key] = {"system_prompt_key": key, "error": str(e)}

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY: System Prompt Modification Detection")
    print("=" * 60)
    print(f"{'Variant':<12} {'Match?':<8} {'Exact Match':<12} {'Avg Prob':<10} {'Avg Margin':<10}")
    print("-" * 60)
    for key in SYSTEM_PROMPTS.keys():
        if key in all_results and "error" not in all_results[key]:
            r = all_results[key]
            match_str = "YES" if r["prompt_match"] else "NO"
            print(
                f"{key:<12} {match_str:<8} {r['exact_match_rate']:.2%}       {r['avg_prob']:.4f}     {r['avg_margin']:.4f}"
            )

    # Save results
    output_path = Path("system_prompt_detection_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
