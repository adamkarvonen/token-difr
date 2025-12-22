"""Demo: Generate with OpenRouter API and verify with Tinker."""

import asyncio
import json
import os
from datetime import datetime
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

from token_difr import (
    TokenSequence,
    compute_metrics_summary,
    construct_prompts,
    encode_thinking_response,
    verify_outputs_tinker,
)

# Default test prompts (used if custom_prompts is None and use_wildchat is False)
DEFAULT_TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms.",
    "Write a haiku about the ocean.",
    "What is 2 + 2?",
    "List three primary colors.",
    "Describe the water cycle.",
    "What causes rainbows?",
    "Explain gravity to a child.",
]

# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
# OPENROUTER_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"

MODEL_NAME = "moonshotai/Kimi-K2-Thinking"
OPENROUTER_MODEL_NAME = "moonshotai/kimi-k2-thinking"


MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
OPENROUTER_MODEL_NAME = "qwen/qwen3-235b-a22b-2507"
PROVIDER = "together"
PROVIDER = "google-vertex"

# Demo configuration
N_PROMPTS = 10  # Use 10 for quick testing, 100 for full run
MAX_TOKENS = 200
MAX_CTX_LEN = 512
USE_WILDCHAT = True  # If True, load prompts from WildChat dataset


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
        Tuple of (content, reasoning) where reasoning may be empty.
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


async def generate_all(
    client: openai.AsyncOpenAI,
    model: str,
    conversations: list[list[dict[str, str]]],
    max_tokens: int,
    temperature: float,
    provider: str,
    concurrency: int = 8,
) -> list[tuple[str, str]]:
    """Generate responses for all conversations concurrently.

    Args:
        client: OpenAI client configured for OpenRouter.
        model: Model name on OpenRouter.
        conversations: List of conversations, where each is a list of message dicts.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        provider: OpenRouter provider name.
        concurrency: Maximum concurrent requests.

    Returns:
        List of (content, reasoning) tuples.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _wrapped(idx: int, messages: list[dict[str, str]]) -> tuple[int, str, str]:
        async with semaphore:
            content, reasoning = await openrouter_request(client, model, messages, max_tokens, temperature, provider)
            return idx, content, reasoning

    tasks = [asyncio.create_task(_wrapped(i, conv)) for i, conv in enumerate(conversations)]
    results: list[tuple[str, str]] = [("", "")] * len(conversations)

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"OpenRouter ({provider})"):
        idx, content, reasoning = await fut
        results[idx] = (content, reasoning)

    return results


def create_token_sequences(
    conversations: list[list[dict[str, str]]],
    responses: list[tuple[str, str]],
    tokenizer,
    max_tokens: int,
) -> list[TokenSequence]:
    """Convert conversations and responses to TokenSequence objects.

    Args:
        conversations: List of conversations, where each is a list of message dicts.
        responses: List of (content, reasoning) tuples.
        tokenizer: HuggingFace tokenizer.
        max_tokens: Maximum number of output tokens.

    Returns:
        List of TokenSequence objects with proper thinking token handling.
    """
    sequences = []
    for conversation, (content, reasoning) in zip(conversations, responses):
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        # Use encode_thinking_response for proper thinking model tokenization
        response_token_ids = encode_thinking_response(content, reasoning, tokenizer, max_tokens)

        sequences.append(
            TokenSequence(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=response_token_ids,
            )
        )
    return sequences


async def run_demo(
    provider: str,
    conversations: list[list[dict[str, str]]],
    tokenizer,
    max_tokens: int = MAX_TOKENS,
    generation_temperature: float = 0.0,
) -> dict:
    """Run the demo for a single provider and return metrics with metadata.

    Args:
        provider: OpenRouter provider name.
        conversations: List of conversations to use as prompts.
        tokenizer: HuggingFace tokenizer for the model.
        max_tokens: Maximum tokens to generate per response.
        generation_temperature: Sampling temperature for OpenRouter (0.0 for greedy).

    Returns:
        Dictionary with metrics and metadata.
    """
    # For greedy verification, use top_k=1 and small temperature
    # Note: Tinker has issues with temperature=0.0, use 1e-8 instead
    top_k = 1
    top_p = 1.0
    verification_temperature = 1e-8
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

    vocab_size = len(tokenizer)

    # Generate responses via OpenRouter
    print(f"\n{'=' * 60}")
    print(f"Provider: {provider}")
    print(f"{'=' * 60}")

    responses = await generate_all(
        openrouter_client,
        OPENROUTER_MODEL_NAME,
        conversations,
        max_tokens=max_tokens,
        temperature=generation_temperature,
        provider=provider,
    )

    # Convert to TokenSequence
    sequences = create_token_sequences(conversations, responses, tokenizer, max_tokens)

    total_tokens = sum(len(s.output_token_ids) for s in sequences)
    print(f"Generated {total_tokens} tokens across {len(sequences)} sequences")

    # Show some sample outputs
    print("\nSample outputs:")
    for i, (conv, (content, reasoning)) in enumerate(zip(conversations[:3], responses[:3])):
        last_user_msg = conv[-1]["content"] if conv else ""
        print(f"  [{i}] {last_user_msg[:40]}...")
        print(f"      → content: {content[:60]}...")
        if reasoning:
            print(f"      → reasoning: {reasoning[:60]}...")

    # Verify with Tinker
    service_client = tinker.ServiceClient(api_key=tinker_api_key)
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    results = verify_outputs_tinker(
        sequences,
        sampling_client=sampling_client,
        vocab_size=vocab_size,
        temperature=verification_temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    summary = compute_metrics_summary(results)

    # Add metadata
    summary["provider"] = provider
    summary["model_name"] = MODEL_NAME
    summary["openrouter_model"] = OPENROUTER_MODEL_NAME
    summary["max_tokens"] = max_tokens
    summary["generation_temperature"] = generation_temperature
    summary["verification_temperature"] = verification_temperature
    summary["top_k"] = top_k
    summary["top_p"] = top_p
    summary["seed"] = seed
    summary["n_prompts"] = len(conversations)
    summary["timestamp"] = datetime.now().isoformat()

    return summary


def sanitize_name(name: str) -> str:
    """Clean up model name for use in filenames."""
    return name.replace("/", "_").replace(".", "_").replace("-", "_")


async def main():
    # providers = ["moonshotai"]  # Testing moonshotai for Kimi-K2
    providers = ["together", "google-vertex"]

    # Load tokenizer once
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Construct prompts
    if USE_WILDCHAT:
        print(f"Loading {N_PROMPTS} prompts from WildChat dataset...")
        conversations = construct_prompts(
            n_prompts=N_PROMPTS,
            max_ctx_len=MAX_CTX_LEN,
            tokenizer=tokenizer,
        )
        print(f"Loaded {len(conversations)} prompts")
    else:
        # Convert simple prompts to conversation format
        conversations = [[{"role": "user", "content": p}] for p in DEFAULT_TEST_PROMPTS]

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
            summary = await run_demo(
                provider=provider,
                conversations=conversations,
                tokenizer=tokenizer,
                max_tokens=MAX_TOKENS,
            )
            all_results[provider] = summary
            print(f"\nResults for {provider}:")
            print(f"  Exact match rate: {summary['exact_match_rate']:.2%}")
            print(f"  Avg probability: {summary['avg_prob']:.4f}")
            print(f"  Avg margin: {summary['avg_margin']:.4f}")
            print(f"  Total tokens: {summary['total_tokens']}")
        except Exception as e:
            import traceback

            print(f"Error with {provider}: {e}")
            traceback.print_exc()
            all_results[provider] = {"provider": provider, "error": str(e)}

        # Save after each provider
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
