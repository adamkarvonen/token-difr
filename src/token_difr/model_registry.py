"""
Model name registry and conversion utilities.

Maps HuggingFace model names to provider-specific model names (Fireworks, OpenRouter).
"""

# HuggingFace name -> Fireworks name (must use accounts/fireworks/models/ prefix)
FIREWORKS_MODEL_REGISTRY: dict[str, str] = {
    # Llama models
    "meta-llama/Llama-3.3-70B-Instruct": "accounts/fireworks/models/llama-v3p3-70b-instruct",
    "meta-llama/Llama-3.1-8B-Instruct": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    "meta-llama/Llama-3.1-70B-Instruct": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    # Kimi models
    "moonshotai/Kimi-K2-Thinking": "accounts/fireworks/models/kimi-k2-thinking",
    "moonshotai/Kimi-K2-Instruct-0905": "accounts/fireworks/models/kimi-k2-instruct-0905",
    # Qwen models
    "Qwen/Qwen2.5-72B-Instruct": "accounts/fireworks/models/qwen2p5-72b-instruct",
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507",
    # Z.ai models
    "zai-org/GLM-4.6": "accounts/fireworks/models/glm-4p6",
    "zai-org/GLM-4.7": "accounts/fireworks/models/glm-4p7",
}

# HuggingFace name -> OpenRouter name (only for models that differ from hf_name.lower())
OPENROUTER_MODEL_REGISTRY: dict[str, str] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen/qwen3-235b-a22b-2507",
    "moonshotai/Kimi-K2-Instruct": "moonshotai/kimi-k2",
    "moonshotai/Kimi-K2-Instruct-0905": "moonshotai/kimi-k2-0905",
    "zai-org/GLM-4.6": "z-ai/glm-4.6",
    "zai-org/GLM-4.7": "z-ai/glm-4.7",
}


def register_fireworks_model(hf_name: str, fireworks_name: str) -> None:
    """
    Register a HuggingFace to Fireworks model name mapping.

    Args:
        hf_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")
        fireworks_name: Fireworks model name (e.g., "accounts/fireworks/models/llama-v3p1-8b-instruct")
    """
    FIREWORKS_MODEL_REGISTRY[hf_name] = fireworks_name


def register_openrouter_model(hf_name: str, openrouter_name: str) -> None:
    """
    Register a HuggingFace to OpenRouter model name mapping.

    Only needed for models where the OpenRouter name differs from hf_name.lower().

    Args:
        hf_name: HuggingFace model name (e.g., "Qwen/Qwen3-235B-A22B-Instruct-2507")
        openrouter_name: OpenRouter model name (e.g., "qwen/qwen3-235b-a22b-2507")
    """
    OPENROUTER_MODEL_REGISTRY[hf_name] = openrouter_name


def get_openrouter_name(hf_name: str) -> str:
    """
    Get the OpenRouter model name for a HuggingFace model.

    If the model is in OPENROUTER_MODEL_REGISTRY, returns the registered name.
    Otherwise, returns hf_name.lower() as the default.

    Args:
        hf_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        OpenRouter model name (e.g., "meta-llama/llama-3.1-8b-instruct")
    """
    if hf_name in OPENROUTER_MODEL_REGISTRY:
        return OPENROUTER_MODEL_REGISTRY[hf_name]
    return hf_name.lower()


def guess_fireworks_name(hf_name: str) -> str:
    """
    Attempt to convert a HuggingFace model name to Fireworks format using heuristics.

    This is a best-effort guess - always verify against actual Fireworks availability.
    The heuristics applied:
    1. Extract model name after the org/ prefix
    2. Convert to lowercase
    3. Replace dots with 'p' (e.g., "3.1" -> "3p1")

    Args:
        hf_name: HuggingFace model name (e.g., "meta-llama/Llama-3.1-8B-Instruct")

    Returns:
        Guessed Fireworks model name (e.g., "accounts/fireworks/models/llama-3p1-8b-instruct")
    """
    # Extract just the model name (after org/)
    if "/" in hf_name:
        model_name = hf_name.split("/", 1)[1]
    else:
        model_name = hf_name

    # Lowercase
    model_name = model_name.lower()

    # Replace dots with 'p'
    model_name = model_name.replace(".", "p")

    return f"accounts/fireworks/models/{model_name}"
