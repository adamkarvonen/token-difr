"""Token verification using Gumbel-Max sampling for LLM outputs."""

from token_difr.api import verify_outputs_api
from token_difr.common import (
    SamplingMethod,
    TokenMetrics,
    TokenSequence,
    compute_metrics_summary,
    construct_prompts,
    encode_thinking_response,
)
from token_difr.local import verify_outputs

__version__ = "0.1.1"

__all__ = [
    "verify_outputs",
    "verify_outputs_api",
    "TokenSequence",
    "TokenMetrics",
    "SamplingMethod",
    "compute_metrics_summary",
    "construct_prompts",
    "encode_thinking_response",
    "__version__",
]
