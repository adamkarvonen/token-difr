"""Tests for OpenRouter API integration and tokenization correctness.

Verifies that our tokenization of OpenRouter responses produces the correct
structure (special tokens, think tags for thinking models, etc.)
"""

import asyncio
import os
from dataclasses import dataclass

import openai
import pytest
from transformers import AutoTokenizer

from token_difr import encode_thinking_response
from token_difr.openrouter_api import openrouter_request

# Load .env file if present
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Expected token structure for Kimi (thinking model)
# Key structural elements: <|im_user|>, <|im_middle|>, <|im_end|>, <|im_assistant|>, <think>, </think>
KIMI_EXPECTED_RESPONSE = [
    "<|im_user|>",
    "user",
    "<|im_middle|>",
    "Hello",
    "<|im_end|>",
    "<|im_assistant|>",
    "assistant",
    "<|im_middle|>",
    "<think>",
    "The",
    " user",
    " has",
    " simply",
    " said",
    ' "',
    "Hello",
    '".',
    " This",
    " is",
    " a",
    " very",
    " basic",
    " greeting",
    ".",
    " I",
    " should",
    " respond",
    " in",
    " a",
    " friendly",
    ",",
    " helpful",
    ",",
    " and",
    " concise",
    " way",
    ".",
    " I",
    " should",
    " ask",
    " how",
    " I",
    " can",
    " assist",
    " them",
    ",",
    " or",
    " offer",
    " some",
    " suggestions",
    " of",
    " what",
    " I",
    " can",
    " do",
    ".\n\n",
    "I'll",
    " keep",
    " it",
    " simple",
    ",",
    " warm",
    ",",
    " and",
    " inviting",
    ".",
    " I",
    " should",
    " be",
    " helpful",
    " and",
    " make",
    " it",
    " easy",
    " for",
    " them",
    " to",
    " continue",
    " the",
    " conversation",
    ".",
    "</think>",
    "Hello",
    "!",
    " How",
    " can",
    " I",
    " help",
    " you",
    " today",
    "?",
    "<|im_end|>",
]

LLAMA_70B_EXPECTED_RESPONSE = [
    "",
    "<|start_header_id|>",
    "system",
    "<|end_header_id|>",
    "\n\n",
    "Cut",
    "ting",
    " Knowledge",
    " Date",
    ":",
    " December",
    " ",
    "202",
    "3",
    "\n",
    "Today",
    " Date",
    ":",
    " ",
    "26",
    " Jul",
    " ",
    "202",
    "4",
    "\n\n",
    "",
    "<|start_header_id|>",
    "user",
    "<|end_header_id|>",
    "\n\n",
    "Hello",
    "",
    "<|start_header_id|>",
    "assistant",
    "<|end_header_id|>",
    "\n\n",
    "Hello",
    "!",
    " How",
    " can",
    " I",
    " assist",
    " you",
    " today",
    "?",
    "",
]


@dataclass
class ModelConfig:
    """Configuration for testing a model."""

    openrouter_model: str  # Model name on OpenRouter
    hf_model: str  # HuggingFace model for tokenizer
    provider: str  # OpenRouter provider
    is_thinking_model: bool  # Whether model uses <think>...</think>
    expected_structure: list[str]  # Expected structural tokens (special tokens, think tags)


MODEL_CONFIGS = {
    "llama": ModelConfig(
        openrouter_model="meta-llama/llama-3.3-70b-instruct",
        hf_model="meta-llama/Llama-3.3-70B-Instruct",
        provider="Fireworks",
        is_thinking_model=False,
        # Llama structure: BOS, system header, system content, user header, user content, assistant header, response
        expected_structure=["<|start_header_id|>", "user", "<|end_header_id|>", "<|start_header_id|>", "assistant", "<|end_header_id|>"],
    ),
    "kimi": ModelConfig(
        openrouter_model="moonshotai/kimi-k2-thinking",
        hf_model="moonshotai/Kimi-K2-Thinking",
        provider="moonshotai",
        is_thinking_model=True,
        # Kimi structure: user marker, middle, content, end, assistant marker, middle, <think>, reasoning, </think>, content, end
        expected_structure=["<|im_user|>", "<|im_middle|>", "<|im_end|>", "<|im_assistant|>", "<|im_middle|>", "<think>", "</think>"],
    ),
}


def get_openrouter_api_key() -> str:
    """Get OpenRouter API key from environment."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY environment variable not set")
    return api_key


@pytest.mark.parametrize("model_key", ["llama", "kimi"])
def test_openrouter_tokenization_structure(model_key: str):
    """Test that OpenRouter responses are tokenized with correct structure.

    Verifies:
    1. openrouter_request returns content (and reasoning for thinking models)
    2. Our tokenization produces correct special token structure
    3. For thinking models, <think>...</think> tags wrap the reasoning
    """
    config = MODEL_CONFIGS[model_key]
    openrouter_api_key = get_openrouter_api_key()

    client = openai.AsyncOpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.hf_model, trust_remote_code=True)
    except Exception as e:
        pytest.skip(f"Could not load {config.hf_model} tokenizer: {e}")

    # Simple test message
    messages = [{"role": "user", "content": "Hello"}]

    # Use openrouter_request from openrouter_api.py
    try:
        content, reasoning = asyncio.run(
            openrouter_request(
                client=client,
                model=config.openrouter_model,
                messages=messages,
                max_tokens=200,
                temperature=0.0,
                provider=config.provider,
            )
        )
    except Exception as e:
        pytest.skip(f"OpenRouter request failed: {e}")

    print(f"\n[{model_key}] OpenRouter response:")
    print(f"  Content: {content[:80]}...")
    if reasoning:
        print(f"  Reasoning: {reasoning[:80]}...")

    # Build full tokenized sequence: prompt + response
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_token_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    response_token_ids = encode_thinking_response(content, reasoning, tokenizer, max_tokens=200)

    # Combine into full sequence
    full_token_ids = prompt_token_ids + response_token_ids
    token_strings = [tokenizer.decode([tid]) for tid in full_token_ids]

    print(f"  Total tokens: {len(full_token_ids)}")

    # Check that expected structural tokens appear in order
    token_str_joined = "".join(token_strings)
    last_pos = -1
    for expected_token in config.expected_structure:
        pos = token_str_joined.find(expected_token, last_pos + 1)
        assert pos > last_pos, (
            f"[{model_key}] Expected '{expected_token}' after position {last_pos}, "
            f"but not found in sequence"
        )
        last_pos = pos

    # For thinking models, verify reasoning is wrapped in think tags
    if config.is_thinking_model:
        assert reasoning, f"[{model_key}] Thinking model should return reasoning"
        decoded_response = tokenizer.decode(response_token_ids)
        assert decoded_response.startswith("<think>"), (
            f"[{model_key}] Response should start with <think>, got: {decoded_response[:50]}"
        )
        assert "</think>" in decoded_response, (
            f"[{model_key}] Response should contain </think>"
        )
        # Verify content appears after </think>
        think_end_pos = decoded_response.find("</think>")
        after_think = decoded_response[think_end_pos + len("</think>"):]
        assert content[:20] in after_think or after_think.strip().startswith(content[:10].strip()), (
            f"[{model_key}] Content should appear after </think>"
        )
        print(f"  Response structure: <think>...reasoning...</think>{content[:30]}...")

    print(f"\n[{model_key}] Tokenization structure test: PASSED")
