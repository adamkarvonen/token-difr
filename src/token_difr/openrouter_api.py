import asyncio
import json
from pathlib import Path

import openai
from transformers import AutoTokenizer
from tqdm import tqdm

from token_difr.common import construct_prompts, encode_thinking_response


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


async def openrouter_request(
    client: openai.AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    provider: str,
    seed: int | None = None,
) -> tuple[str, str]:
    """Make a single OpenRouter request.

    Returns:
        Tuple of (content, reasoning) where reasoning may be empty for non-thinking models.
    """
    extra_body: dict = {
        "provider": {"order": [provider]},
    }

    completion = await client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        extra_body=extra_body,
    )
    content = completion.choices[0].message.content or ""
    reasoning = getattr(completion.choices[0].message, "reasoning", None) or ""
    return content, reasoning


async def run_all_prompts(
    client: openai.AsyncOpenAI,
    model: str,
    prompts: list[list[dict[str, str]]],
    max_tokens: int,
    temperature: float,
    provider: str,
    concurrency: int = 8,
) -> list[tuple[list[dict[str, str]], str, str]]:
    """Generate responses for all prompts concurrently.

    Returns:
        List of (prompt, content, reasoning) tuples.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _wrapped(idx: int, prompt: list[dict[str, str]]) -> tuple[int, str, str]:
        async with semaphore:
            content, reasoning = await openrouter_request(
                client,
                model,
                prompt,
                max_tokens,
                temperature=temperature,
                provider=provider,
            )
            return idx, content, reasoning

    tasks = [asyncio.create_task(_wrapped(i, p)) for i, p in enumerate(prompts)]
    results: list[tuple[list[dict[str, str]], str, str]] = [([], "", "") for _ in prompts]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"OpenRouter ({provider})"):
        idx, content, reasoning = await fut
        results[idx] = (prompts[idx], content, reasoning)

    return results


def save_results(
    samples: list[tuple[list[dict[str, str]], str, str]],
    save_path: Path,
    config: dict[str, object],
    model_name: str,
    max_tokens: int,
) -> None:
    """Save samples as JSON in VLLM-style format with tokenized prompts and responses.

    Args:
        samples: List of (prompt, content, reasoning) tuples.
        save_path: Path to save the JSON file.
        config: Configuration dictionary to include in output.
        model_name: HuggingFace model name for tokenizer.
        max_tokens: Maximum tokens for response truncation.
    """
    # Load tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Tokenize prompts and responses
    vllm_samples = []
    for prompt, content, reasoning in tqdm(samples, desc="Tokenizing samples"):
        # Tokenize prompt (conversation array)
        rendered_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False, return_tensors=None)

        # Tokenize response using encode_thinking_response for proper handling of thinking models
        response_token_ids = encode_thinking_response(content, reasoning, tokenizer, max_tokens)

        # Create VLLM-style sample
        vllm_samples.append({"prompt_token_ids": prompt_token_ids, "outputs": [{"token_ids": response_token_ids}]})

    del tokenizer

    # Save as JSON
    payload = {"config": config, "samples": vllm_samples}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Saved {len(samples)} samples to {save_path}")


async def main():
    model_name = "meta-llama/llama-3.1-8b-instruct"
    max_tokens = 500
    temperature = 0.0
    concurrency = 50

    for provider in ["cerebras", "hyperbolic", "groq", "siliconflow/fp8", "deepinfra"]:
        save_dir = Path("openrouter_responses")
        n_samples = 2000
        max_ctx_len = 512

        # Load OpenRouter API key from file to keep the script simple.
        with open("openrouter_api_key.txt") as f:
            api_key = f.read().strip()

        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        prompts = construct_prompts(n_prompts=n_samples, max_ctx_len=max_ctx_len, model_name=model_name)
        print(f"Loaded {len(prompts)} prompts from dataset.")

        responses = await run_all_prompts(
            client,
            model_name,
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            provider=provider,
            concurrency=concurrency,
        )

        save_dir.mkdir(parents=True, exist_ok=True)
        model_tag = _sanitize(f"{provider}_{model_name}")
        save_filename = f"openrouter_{model_tag}_token_difr_prompts_test.json"
        config = {
            "model": model_name,
            "provider": provider,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n_samples": n_samples,
            "max_ctx_len": max_ctx_len,
        }
        save_results(
            responses,
            save_dir / save_filename,
            config=config,
            model_name=model_name,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    asyncio.run(main())
