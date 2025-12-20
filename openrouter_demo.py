"""Demo: Generate with OpenRouter API and verify with Tinker."""

import asyncio
import json
import os
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import openai
import tinker
from transformers import AutoTokenizer
from tqdm import tqdm

from token_difr import TokenSequence, compute_metrics_summary, verify_outputs_tinker

# Same test prompts as test_tinker.py
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

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct"


async def openrouter_request(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    provider: str,
) -> str:
    """Make a single OpenRouter request."""
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body={
            "provider": {"order": [provider]},
        },
    )
    return completion.choices[0].message.content or ""


async def generate_all(
    client: openai.AsyncOpenAI,
    model: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    provider: str,
    concurrency: int = 8,
) -> list[str]:
    """Generate responses for all prompts concurrently."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _wrapped(idx: int, prompt: str) -> tuple[int, str]:
        async with semaphore:
            messages = [{"role": "user", "content": prompt}]
            response = await openrouter_request(
                client, model, messages, max_tokens, temperature, provider
            )
            return idx, response

    tasks = [asyncio.create_task(_wrapped(i, p)) for i, p in enumerate(prompts)]
    results: list[str] = [""] * len(prompts)

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"OpenRouter ({provider})"):
        idx, response = await fut
        results[idx] = response

    return results


def create_token_sequences(
    prompts: list[str],
    responses: list[str],
    tokenizer,
    max_tokens: int,
) -> list[TokenSequence]:
    """Convert prompts and responses to TokenSequence objects."""
    sequences = []
    for prompt, response in zip(prompts, responses):
        messages = [{"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        response_token_ids = tokenizer.encode(response, add_special_tokens=False)[:max_tokens]

        sequences.append(TokenSequence(
            prompt_token_ids=prompt_token_ids,
            output_token_ids=response_token_ids,
        ))
    return sequences


async def run_demo(provider: str) -> dict:
    """Run the demo for a single provider and return metrics."""
    max_tokens = 100
    temperature = 0.0  # Greedy for verification
    top_k = 20
    top_p = 0.95
    seed = 42

    # Load API keys
    openrouter_key_path = Path("openrouter_api_key.txt")
    if openrouter_key_path.exists():
        openrouter_api_key = openrouter_key_path.read_text().strip()
    else:
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    tinker_api_key = os.environ.get("TINKER_API_KEY", "")

    if not openrouter_api_key:
        raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY or create openrouter_api_key.txt")
    if not tinker_api_key:
        raise ValueError("Tinker API key not found. Set TINKER_API_KEY environment variable")

    # Initialize clients
    openrouter_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Generate responses via OpenRouter
    print(f"\n{'='*60}")
    print(f"Provider: {provider}")
    print(f"{'='*60}")

    responses = await generate_all(
        openrouter_client,
        OPENROUTER_MODEL,
        TEST_PROMPTS,
        max_tokens=max_tokens,
        temperature=temperature,
        provider=provider,
    )

    # Convert to TokenSequence
    sequences = create_token_sequences(TEST_PROMPTS, responses, tokenizer, max_tokens)

    total_tokens = sum(len(s.output_token_ids) for s in sequences)
    print(f"Generated {total_tokens} tokens across {len(sequences)} sequences")

    # Show some sample outputs
    print("\nSample outputs:")
    for i, (prompt, response) in enumerate(zip(TEST_PROMPTS[:3], responses[:3])):
        print(f"  [{i}] {prompt[:40]}...")
        print(f"      → {response[:80]}...")

    # Verify with Tinker
    service_client = tinker.ServiceClient(api_key=tinker_api_key)
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    results = verify_outputs_tinker(
        sequences,
        sampling_client=sampling_client,
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    summary = compute_metrics_summary(results)
    summary["provider"] = provider

    return summary


def sanitize_name(name: str) -> str:
    """Clean up model name for use in filenames."""
    return name.replace("/", "_").replace(".", "_").replace("-", "_")


async def main():
    providers = ["Groq", "SiliconFlow", "Cerebras"]

    # Create output filename based on model name
    model_tag = sanitize_name(MODEL_NAME)
    output_path = Path(f"openrouter_demo_{model_tag}.json")

    # Load existing results if file exists
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    for provider in providers:
        try:
            summary = await run_demo(provider)
            all_results[provider] = summary
            print(f"\nResults for {provider}:")
            print(f"  Exact match rate: {summary['exact_match_rate']:.2%}")
            print(f"  Avg probability: {summary['avg_prob']:.4f}")
            print(f"  Avg margin: {summary['avg_margin']:.4f}")
        except Exception as e:
            print(f"Error with {provider}: {e}")
            all_results[provider] = {"provider": provider, "error": str(e)}

        # Save after each provider
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
