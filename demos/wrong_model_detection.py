"""Test detection of model version mismatch via Gumbel-Max verification.

Experiment:
- Generate responses via OpenRouter with one model version
- Verify via Fireworks with a different model version
- Measure how well we can detect when a different model version was used
"""

import asyncio
import json
import os
from dataclasses import dataclass
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


@dataclass
class ModelPair:
    """A pair of models for generation and verification."""

    generation_model: str  # HuggingFace model name for generation
    verification_model: str  # HuggingFace model name for verification
    provider: str  # OpenRouter provider
    name: str  # Display name for this pair


# Model pairs to test
# Note: Only kimi-k2-instruct-0905 is available on Fireworks for verification
MODEL_PAIRS = [
    # Mismatch: generate with original, verify with 0905
    ModelPair(
        name="glm_v1_gen_v2_verify",
        generation_model="zai-org/GLM-4.6",
        verification_model="zai-org/GLM-4.7",
        provider="z-ai",
    ),
    # Control: same model (0905 for both)
    ModelPair(
        name="glm_v2_control",
        generation_model="zai-org/GLM-4.7",
        verification_model="zai-org/GLM-4.7",
        provider="z-ai",
    ),
]

# Demo configuration
N_PROMPTS = 50
MAX_TOKENS = 200
MAX_CTX_LEN = 512
SYSTEM_PROMPT = "You are a helpful assistant."

# Output directory
OUTPUT_DIR = Path("audit_results")


async def run_experiment(
    model_pair: ModelPair,
    conversations: list[list[dict[str, str]]],
    tokenizer,
    openrouter_client: AsyncOpenAI,
    fireworks_client: AsyncOpenAI,
    vocab_size: int,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Run experiment for a single model pair."""
    # Verification parameters
    top_k = 50
    top_p = 0.95
    seed = 42
    temperature = 0.0
    concurrency = 10

    # Get model names from registry
    openrouter_model = get_openrouter_name(model_pair.generation_model)
    fireworks_model = FIREWORKS_MODEL_REGISTRY[model_pair.verification_model]

    print(f"\n{'=' * 60}")
    print(f"Model Pair: {model_pair.name}")
    print(f"  Generation: {openrouter_model} (via {model_pair.provider})")
    print(f"  Verification: {fireworks_model}")
    print(f"{'=' * 60}")

    # Generate responses via OpenRouter
    responses = await generate_openrouter_responses(
        client=openrouter_client,
        conversations=conversations,
        model=openrouter_model,
        provider=model_pair.provider,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        concurrency=concurrency,
    )

    # Tokenize responses
    sequences = tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens)

    total_tokens = sum(len(s.output_token_ids) for s in sequences)
    print(f"Generated {total_tokens} tokens across {len(sequences)} sequences")

    # Show sample outputs
    print("\nSample outputs:")
    for i, (conv, resp) in enumerate(zip(conversations[:2], responses[:2])):
        last_user_msg = conv[-1]["content"] if conv else ""
        content = resp.choices[0].message.content or ""
        print(f"  [{i}] {last_user_msg[:50]}...")
        print(f"      → {content[:60]}...")

    # Verify with Fireworks
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
        concurrency=concurrency,
    )

    summary = compute_metrics_summary(results)

    # Add metadata
    summary["pair_name"] = model_pair.name
    summary["generation_model"] = model_pair.generation_model
    summary["generation_provider"] = model_pair.provider
    summary["verification_model"] = model_pair.verification_model
    summary["models_match"] = model_pair.generation_model == model_pair.verification_model
    summary["max_tokens"] = max_tokens
    summary["n_prompts"] = len(conversations)
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

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Initialize clients
    openrouter_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )
    fireworks_client = AsyncOpenAI(
        api_key=fireworks_api_key,
        base_url="https://api.fireworks.ai/inference/v1",
    )

    # Load tokenizer (same for both Kimi versions)
    hf_model = MODEL_PAIRS[0].generation_model
    print(f"Loading tokenizer for {hf_model}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Load prompts
    print(f"Loading {N_PROMPTS} prompts from WildChat dataset...")
    conversations = construct_prompts(
        n_prompts=N_PROMPTS,
        max_ctx_len=MAX_CTX_LEN,
        tokenizer=tokenizer,
        system_prompt=SYSTEM_PROMPT,
    )
    print(f"Loaded {len(conversations)} prompts")

    # Run experiments for each model pair
    all_results = {}

    for model_pair in MODEL_PAIRS:
        try:
            summary = await run_experiment(
                model_pair=model_pair,
                conversations=conversations,
                tokenizer=tokenizer,
                openrouter_client=openrouter_client,
                fireworks_client=fireworks_client,
                vocab_size=vocab_size,
            )
            all_results[model_pair.name] = summary

            print(f"\nResults for {model_pair.name}:")
            print(f"  Models match: {summary['models_match']}")
            print(f"  Exact match rate: {summary['exact_match_rate']:.2%}")
            print(f"  Avg probability: {summary['avg_prob']:.4f}")
            print(f"  Avg margin: {summary['avg_margin']:.4f}")

        except Exception as e:
            import traceback

            print(f"Error with {model_pair.name}: {e}")
            traceback.print_exc()
            all_results[model_pair.name] = {"pair_name": model_pair.name, "error": str(e)}

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: Wrong Model Detection")
    print("=" * 70)
    print(f"{'Pair Name':<25} {'Match?':<8} {'Exact Match':<12} {'Avg Prob':<10} {'Avg Margin':<10}")
    print("-" * 70)
    for model_pair in MODEL_PAIRS:
        if model_pair.name in all_results and "error" not in all_results[model_pair.name]:
            r = all_results[model_pair.name]
            match_str = "YES" if r["models_match"] else "NO"
            print(f"{model_pair.name:<25} {match_str:<8} {r['exact_match_rate']:.2%}       ", end="")
            print(f"{r['avg_prob']:.4f}     {r['avg_margin']:.4f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"wrong_model_detection_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
