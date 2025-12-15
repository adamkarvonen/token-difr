"""Token verification using Gumbel-Max sampling for LLM outputs."""

from token_difr.verify import (
    SamplingMethod,
    TokenMetrics,
    TokenSequence,
    verify_outputs,
)

__version__ = "0.1.0"

__all__ = [
    "verify_outputs",
    "TokenSequence",
    "TokenMetrics",
    "SamplingMethod",
    "__version__",
]
