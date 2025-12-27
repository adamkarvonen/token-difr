# %%
"""Simple OpenRouter test script - run inference on prompts."""

import asyncio
import os
from pathlib import Path
from transformers import AutoTokenizer

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import openai

# %%
# Configuration - edit these values
PROVIDER = "Groq"  # e.g., "Groq", "SiliconFlow", "Cerebras", "moonshotai"
# PROVIDER = "together"
MODEL = "meta-llama/llama-3.1-8b-instruct"  # OpenRouter model name
# MODEL = "meta-llama/llama-3.3-70b-instruct"
# MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# MODEL = "moonshotai/Kimi-K2-Thinking"
# PROVIDER = "moonshotai"

PROMPTS = [
    "How should I boil an egg?",
    # "Explain photosynthesis in simple terms.",
    # "Write a haiku about the ocean.",
]

PROMPTS = PROMPTS * 10

# %%
# Load API key
openrouter_key_path = Path("openrouter_api_key.txt")
if openrouter_key_path.exists():
    api_key = openrouter_key_path.read_text().strip()
else:
    api_key = os.environ.get("OPENROUTER_API_KEY", "")

if not api_key:
    raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY or create openrouter_api_key.txt")

# %%
# Create client
client = openai.AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)


tokenizer = AutoTokenizer.from_pretrained(MODEL)
prompt = "How should I boil an egg?"
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True, return_tensors=None)


# %%
async def run_inference(prompt: str) -> openai.ChatCompletion:
    """Run inference on a single prompt."""
    completion = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=40,
        temperature=1.0,
        seed=42,
        logprobs=True,
        top_logprobs=1,
        # echo=True,
        extra_body={
            # "echo": True,
            "provider": {"only": [PROVIDER]},
            # "debug": {"echo_upstream_body": True},
            # "top_logprobs": 1,
            # "logprobs": True,
        },
        # extra_body={"provider": {"order": [PROVIDER]}, "logprobs": 20},
    )

    # Use completions instead of chat.completions
    # completion = await client.completions.create(
    #     model=MODEL,
    #     # prompt="How should I boil an egg?",
    #     prompt=prompt_tokens,
    #     max_tokens=10,
    #     echo=True,
    #     logprobs=5,
    #     extra_body={"echo": True, "provider": {"order": [PROVIDER]}, "debug": {"echo_upstream_body": True}},
    # )

    return completion


async def run_all(prompts: list[str]) -> list[str]:
    """Run inference on all prompts concurrently."""
    tasks = [run_inference(p) for p in prompts]
    return await asyncio.gather(*tasks)


# %%
# Run inference
responses = await run_all(PROMPTS)
# %%
print(f"Provider: {PROVIDER}")
print(f"Model: {MODEL}")
print()
for prompt, response in zip(PROMPTS, responses):
    print(f"Prompt: {prompt}")
    print(f"Response: {response.choices[0].message.content}\n")

# %%
first_response = responses[0]

print(first_response)

# %%
