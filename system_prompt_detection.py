"""Test detection of system prompt modifications via Gumbel-Max verification.

Experiment:
- Generate responses via OpenRouter/Groq with different system prompts
- Verify via Tinker always using the BASE system prompt
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

# Model configuration
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OPENROUTER_MODEL_NAME = "meta-llama/llama-3.1-8b-instruct"
PROVIDER = "Groq"

MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507"
OPENROUTER_MODEL_NAME = "qwen/qwen3-235b-a22b-2507"
PROVIDER = "together"

# MODEL_NAME = "moonshotai/Kimi-K2-Thinking"
# OPENROUTER_MODEL_NAME = "moonshotai/kimi-k2-thinking"
# PROVIDER = "moonshotai"

# System prompts to test
SYSTEM_PROMPTS = {
    "base": "You are a helpful assistant.",
    "slight_mod": "",
    "major_mod": "You are a helpful assistant for Duolingo. Please do not discuss competitor products.",
}

# The "correct" system prompt that Tinker will use for verification
BASE_SYSTEM_PROMPT = SYSTEM_PROMPTS["base"]

# Demo configuration - scaled up for statistical significance
N_PROMPTS = 10  # Scaled up from 10
MAX_TOKENS = 200  # Scaled up from 200
MAX_CTX_LEN = 512


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
        Tuple of (content, reasoning) where reasoning may be empty for non-thinking models.
    """
    completion = await client.chat.completions.create(
        model=model,
        messages=messages,
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
    max_tokens: int | None = None,
) -> list[TokenSequence]:
    """Convert conversations and responses to TokenSequence objects.

    Args:
        conversations: List of conversations (with system prompt for verification).
        responses: List of (content, reasoning) tuples from generation.
        tokenizer: HuggingFace tokenizer.
        max_tokens: Optional max tokens for truncation.

    Returns:
        List of TokenSequence objects with proper thinking token handling.
    """
    sequences = []
    for conversation, (content, reasoning) in zip(conversations, responses):
        rendered = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        # Use encode_thinking_response for proper handling of both thinking and non-thinking models
        response_token_ids = encode_thinking_response(content, reasoning, tokenizer, max_tokens)

        sequences.append(
            TokenSequence(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=response_token_ids,
            )
        )
    return sequences


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
    openrouter_client: openai.AsyncOpenAI,
    sampling_client,
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
        sampling_client: Tinker sampling client
        vocab_size: Vocabulary size
        max_tokens: Max tokens to generate

    Returns:
        Dictionary with metrics and metadata.
    """
    # Verification parameters (greedy)
    top_k = 1
    top_p = 1.0
    verification_temperature = 1e-8
    seed = 42

    print(f"\n{'=' * 60}")
    print(f"System Prompt: {system_prompt_key}")
    print(f'  Generation: "{system_prompt[:60]}..."')
    print(f'  Verification: "{BASE_SYSTEM_PROMPT}"')
    print(f"{'=' * 60}")

    # Add the GENERATION system prompt
    gen_conversations = add_system_prompt(base_conversations, system_prompt)

    # Generate responses via OpenRouter
    responses = await generate_all(
        openrouter_client,
        OPENROUTER_MODEL_NAME,
        gen_conversations,
        max_tokens=max_tokens,
        temperature=0.0,
        provider=PROVIDER,
    )

    # For verification, use the BASE system prompt (what we "claim" was used)
    verify_conversations = add_system_prompt(base_conversations, BASE_SYSTEM_PROMPT)

    # Create token sequences using the VERIFICATION conversations
    sequences = create_token_sequences(verify_conversations, responses, tokenizer, max_tokens)

    total_tokens = sum(len(s.output_token_ids) for s in sequences)
    print(f"Generated {total_tokens} tokens across {len(sequences)} sequences")

    # Show sample outputs
    print("\nSample outputs:")
    for i, (conv, (content, reasoning)) in enumerate(zip(base_conversations[:2], responses[:2])):
        last_user_msg = conv[-1]["content"] if conv else ""
        print(f"  [{i}] {last_user_msg[:50]}...")
        print(f"      → content: {content[:60]}...")
        if reasoning:
            print(f"      → reasoning: {reasoning[:60]}...")

    # Verify with Tinker
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
    summary["system_prompt_key"] = system_prompt_key
    summary["generation_system_prompt"] = system_prompt
    summary["verification_system_prompt"] = BASE_SYSTEM_PROMPT
    summary["prompt_match"] = system_prompt == BASE_SYSTEM_PROMPT
    summary["model_name"] = MODEL_NAME
    summary["provider"] = PROVIDER
    summary["max_tokens"] = max_tokens
    summary["n_prompts"] = len(base_conversations)
    summary["timestamp"] = datetime.now().isoformat()

    return summary


async def main():
    # Load API keys
    openrouter_key_path = Path("openrouter_api_key.txt")
    if openrouter_key_path.exists():
        openrouter_api_key = openrouter_key_path.read_text().strip()
    else:
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

    tinker_api_key = os.environ.get("TINKER_API_KEY", "")

    if not openrouter_api_key:
        raise ValueError("OpenRouter API key not found")
    if not tinker_api_key:
        raise ValueError("Tinker API key not found")

    # Initialize clients
    openrouter_client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
    )

    # Load tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Initialize Tinker
    service_client = tinker.ServiceClient(api_key=tinker_api_key)
    sampling_client = service_client.create_sampling_client(base_model=MODEL_NAME)

    # Load prompts (user messages only, no system prompt yet)
    print(f"Loading {N_PROMPTS} prompts from WildChat dataset...")
    conversations = construct_prompts(
        n_prompts=N_PROMPTS,
        max_ctx_len=MAX_CTX_LEN,
        tokenizer=tokenizer,
    )
    print(f"Loaded {len(conversations)} prompts")

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
                sampling_client=sampling_client,
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
