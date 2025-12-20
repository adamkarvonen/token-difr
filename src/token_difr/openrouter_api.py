import asyncio
import json
from pathlib import Path

import openai
from transformers import AutoTokenizer
from tqdm import tqdm

from token_difr_vllm import AttestationConfig, construct_dataset as construct_dataset_vllm


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


async def openrouter_request(
    client: openai.AsyncOpenAI,
    model: str,
    prompt: list[dict[str, str]],
    max_completion_tokens: int,
    temperature: float,
    provider: str,
) -> tuple[list[dict[str, str]], str]:
    completion = await client.chat.completions.create(
        model=model,
        messages=prompt,  # type: ignore[arg-type]
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        extra_body={
            "provider": {"order": [provider]},
            "reasoning": {
                "enabled": False,
            },
        },
    )
    text = completion.choices[0].message.content or ""
    return prompt, text


async def run_all_prompts(
    client: openai.AsyncOpenAI,
    api_llm: str,
    prompts: list[list[dict[str, str]]],
    max_completion_tokens: int,
    temperature: float,
    provider: str,
    concurrency: int = 8,
) -> list[tuple[list[dict[str, str]], str]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _wrapped(idx: int, prompt: list[dict[str, str]]) -> tuple[int, list[dict[str, str]], str]:
        async with semaphore:
            p, r = await openrouter_request(
                client,
                api_llm,
                prompt,
                max_completion_tokens,
                temperature=temperature,
                provider=provider,
            )
            return idx, p, r

    tasks = [asyncio.create_task(_wrapped(i, p)) for i, p in enumerate(prompts)]
    results: list[tuple[list[dict[str, str]], str]] = [([], "") for _ in prompts]

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="OpenRouter requests"):
        idx, prompt, response = await fut
        results[idx] = (prompt, response)

    return results


def construct_dataset(cfg: AttestationConfig) -> list[list[dict[str, str]]]:
    """Import and use construct_dataset from token_difr_vllm, returning only conversation prompts."""
    _, _, conversation_prompts = construct_dataset_vllm(cfg)
    return conversation_prompts


def save_results(
    samples: list[tuple[list[dict[str, str]], str]],
    save_path: Path,
    config: dict[str, object],
    model_name: str,
    max_completion_tokens: int,
) -> None:
    """Save samples as JSON in VLLM-style format with tokenized prompts and responses."""
    # Load tokenizer for the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize prompts and responses
    vllm_samples = []
    for prompt, response in tqdm(samples, desc="Tokenizing samples"):
        # Tokenize prompt (conversation array)
        rendered_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False, return_tensors=None)

        # Tokenize response (plain string)
        response_token_ids = tokenizer.encode(response, add_special_tokens=False, return_tensors=None)
        response_token_ids = response_token_ids[:max_completion_tokens]

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
    max_completion_tokens = 500
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

        prompts = construct_dataset(
            AttestationConfig(n_samples=n_samples, max_ctx_len=max_ctx_len, trusted_model_name=model_name)
        )
        print(f"Loaded {len(prompts)} prompts from dataset.")

        responses = await run_all_prompts(
            client,
            model_name,
            prompts,
            max_completion_tokens=max_completion_tokens,
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
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
            "n_samples": n_samples,
            "max_ctx_len": max_ctx_len,
        }
        save_results(
            responses,
            save_dir / save_filename,
            config=config,
            model_name=model_name,
            max_completion_tokens=max_completion_tokens,
        )


if __name__ == "__main__":
    asyncio.run(main())
