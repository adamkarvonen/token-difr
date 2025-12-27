"""Token verification using Gumbel-Max sampling for LLM outputs."""

import os

# Disable HuggingFace tokenizers parallelism to avoid fork warnings.
# The parallelism is for CPU-bound batch tokenization; our bottleneck is API I/O.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from token_difr.api import verify_outputs_fireworks, verify_outputs_tinker
from token_difr.audit import AuditResult, audit_provider
from token_difr.common import (
    SamplingMethod,
    TokenMetrics,
    TokenSequence,
    compute_metrics_summary,
    construct_prompts,
    encode_thinking_response,
)
from token_difr.model_registry import (
    FIREWORKS_MODEL_REGISTRY,
    OPENROUTER_MODEL_REGISTRY,
    get_openrouter_name,
    guess_fireworks_name,
    register_fireworks_model,
    register_openrouter_model,
)
from token_difr.openrouter_api import tokenize_openrouter_responses
from token_difr.local import verify_outputs

__version__ = "0.1.2"

__all__ = [
    # High-level audit API
    "audit_provider",
    "AuditResult",
    "construct_prompts",
    # Low-level verification
    "verify_outputs",
    "verify_outputs_fireworks",
    "verify_outputs_tinker",
    "tokenize_openrouter_responses",
    "TokenSequence",
    "TokenMetrics",
    "SamplingMethod",
    "compute_metrics_summary",
    "encode_thinking_response",
    # Model registry
    "FIREWORKS_MODEL_REGISTRY",
    "OPENROUTER_MODEL_REGISTRY",
    "get_openrouter_name",
    "guess_fireworks_name",
    "register_fireworks_model",
    "register_openrouter_model",
    "__version__",
]
