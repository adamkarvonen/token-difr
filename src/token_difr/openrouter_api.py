import asyncio
import json
import os
from pathlib import Path

import openai
from openai.types.chat import ChatCompletion
from transformers import AutoTokenizer
from tqdm import tqdm

from token_difr.common import TokenSequence, construct_prompts, encode_thinking_response


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


async def generate_openrouter_responses(
    client: openai.AsyncOpenAI,
    conversations: list[list[dict[str, str]]],
    model: str,
    provider: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
    concurrency: int = 8,
    seed: int | None = None,
) -> list[ChatCompletion]:
    """Generate responses for multiple conversations via OpenRouter.

    Args:
        client: OpenRouter AsyncOpenAI client.
        conversations: List of conversations (each is a list of message dicts).
        model: OpenRouter model name.
        provider: Backend provider (e.g., "fireworks", "cerebras").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        concurrency: Number of concurrent requests.
        seed: Optional random seed.

    Returns:
        List of raw ChatCompletion objects in the same order as input conversations.
    """
    semaphore = asyncio.Semaphore(concurrency)
    extra_body: dict = {"provider": {"only": [provider]}}

    async def _request(idx: int, messages: list[dict[str, str]]) -> tuple[int, ChatCompletion]:
        async with semaphore:
            completion = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                extra_body=extra_body,
            )
            return idx, completion

    tasks = [asyncio.create_task(_request(i, conv)) for i, conv in enumerate(conversations)]
    results: list[ChatCompletion | None] = [None] * len(conversations)

    for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"OpenRouter ({provider})"):
        idx, completion = await fut
        results[idx] = completion

    return results  # type: ignore[return-value]


def tokenize_openrouter_responses(
    conversations: list[list[dict[str, str]]],
    responses: list[ChatCompletion],
    tokenizer,
    max_tokens: int | None = None,
) -> list[TokenSequence]:
    """Convert OpenRouter responses to TokenSequence objects.

    Args:
        conversations: List of input conversations (for prompt tokenization).
        responses: List of raw ChatCompletion objects from generate_openrouter_responses.
        tokenizer: HuggingFace tokenizer for the model.
        max_tokens: Optional maximum tokens for response truncation.

    Returns:
        List of TokenSequence objects.
    """
    outputs = []

    for conv, completion in zip(conversations, responses, strict=True):
        # Extract content and reasoning from response
        message = completion.choices[0].message
        content = message.content or ""
        # reasoning is an OpenRouter extension - check both direct attribute and model_extra
        reasoning = getattr(message, "reasoning", None)
        reasoning = reasoning or ""

        # Tokenize prompt
        rendered = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        prompt_token_ids = tokenizer.encode(rendered, add_special_tokens=False)

        # Tokenize response
        response_token_ids = encode_thinking_response(content, reasoning, tokenizer, max_tokens)

        outputs.append(TokenSequence(prompt_token_ids=prompt_token_ids, output_token_ids=response_token_ids))

    return outputs


def save_results(
    conversations: list[list[dict[str, str]]],
    responses: list[ChatCompletion],
    save_path: Path,
    config: dict[str, object],
    model_name: str,
    max_tokens: int,
) -> None:
    """Save responses as JSON in VLLM-style format with tokenized prompts and responses.

    Args:
        conversations: List of input conversations.
        responses: List of raw ChatCompletion objects.
        save_path: Path to save the JSON file.
        config: Configuration dictionary to include in output.
        model_name: HuggingFace model name for tokenizer.
        max_tokens: Maximum tokens for response truncation.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    outputs = tokenize_openrouter_responses(conversations, responses, tokenizer, max_tokens)
    del tokenizer

    vllm_samples = [
        {"prompt_token_ids": seq.prompt_token_ids, "outputs": [{"token_ids": seq.output_token_ids}]} for seq in outputs
    ]

    payload = {"config": config, "samples": vllm_samples}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Saved {len(outputs)} samples to {save_path}")


async def main():
    model_name = "meta-llama/llama-3.1-8b-instruct"
    max_tokens = 500
    temperature = 0.0
    concurrency = 50

    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    for provider in ["cerebras", "hyperbolic", "groq", "siliconflow/fp8", "deepinfra"]:
        save_dir = Path("openrouter_responses")
        n_samples = 2000
        max_ctx_len = 512

        client = openai.AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
        )

        conversations = construct_prompts(n_prompts=n_samples, max_ctx_len=max_ctx_len, model_name=model_name)
        print(f"Loaded {len(conversations)} prompts from dataset.")

        responses = await generate_openrouter_responses(
            client,
            conversations,
            model_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
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
            conversations,
            responses,
            save_dir / save_filename,
            config=config,
            model_name=model_name,
            max_tokens=max_tokens,
        )


if __name__ == "__main__":
    asyncio.run(main())
