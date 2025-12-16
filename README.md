# token-difr

Verify LLM outputs using Gumbel-Max sampling verification.

## Installation

```bash
pip install token-difr
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- vLLM >= 0.10.1
- CUDA-capable GPU

## Quick Start

```python
from token_difr import verify_outputs, TokenSequence

# Create token sequences to verify
# (typically from an untrusted LLM provider)
outputs = [
    TokenSequence(
        prompt_token_ids=[128000, 2323, 374, 264, 1296],
        output_token_ids=[264, 1296, 13, 578, 4320]
    )
]

# Verify the outputs against a trusted model
# All sampling parameters must match what was used during generation
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=1.0,  # Required: must match generation
    top_k=50,         # Required: must match generation
    top_p=0.95,       # Required: must match generation
    seed=42,          # Required: must match generation
)

# Check verification results
for seq_idx, token_metrics in enumerate(results):
    for tok_idx, metrics in enumerate(token_metrics):
        print(f"Seq {seq_idx}, Token {tok_idx}:")
        print(f"  exact_match: {metrics.exact_match}")
        print(f"  prob: {metrics.prob:.4f}")
        print(f"  margin: {metrics.margin:.4f}")
        print(f"  logit_rank: {metrics.logit_rank}")
        print(f"  gumbel_rank: {metrics.gumbel_rank}")
```

## API Reference

### `verify_outputs`

```python
def verify_outputs(
    outputs: list[TokenSequence],
    model_name: str,
    *,
    temperature: float,      # Required
    top_k: int,              # Required
    top_p: float,            # Required
    seed: int,               # Required
    max_model_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    vllm_kwargs: dict[str, Any] | None = None,
    sampling_method: SamplingMethod = SamplingMethod.VLLM_GUMBEL_MAX,
    gpu_memory_utilization: float = 0.7,
    verbose: bool = True,
) -> list[list[TokenMetrics]]:
```

**Parameters:**

- `outputs`: List of `TokenSequence` objects containing prompt and output token IDs
- `model_name`: HuggingFace model name (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`)
- `temperature`: Sampling temperature used during generation. **Required.**
- `top_k`: Top-k sampling parameter. **Required.**
- `top_p`: Top-p (nucleus) sampling parameter. **Required.**
- `seed`: Random seed used during generation. **Required.**
- `max_model_len`: Maximum model context length. If None, auto-computed from outputs
- `dtype`: Model dtype, e.g., `torch.bfloat16`, `torch.float16` (default: `torch.bfloat16`)
- `vllm_kwargs`: Additional kwargs for vLLM's LLM constructor (e.g., for quantization)
- `sampling_method`: Sampling method to verify against (default: `VLLM_GUMBEL_MAX`)
- `gpu_memory_utilization`: Fraction of GPU memory to use (default: 0.7)
- `verbose`: Whether to show progress and print summary (default: True)

**Returns:**

List of lists of `TokenMetrics`, one per token in each output sequence.

### `TokenSequence`

```python
@dataclass
class TokenSequence:
    prompt_token_ids: list[int]
    output_token_ids: list[int]
```

### `TokenMetrics`

```python
@dataclass
class TokenMetrics:
    exact_match: bool   # Whether token matches under verification
    prob: float         # Probability of the actual token
    margin: float       # Margin between max and actual token scores
    logit_rank: float   # Rank of actual token by logit value
    gumbel_rank: float  # Rank of actual token by Gumbel score
```

### `SamplingMethod`

```python
class SamplingMethod(Enum):
    VLLM_GUMBEL_MAX = "vllm_gumbel_max"
```

### `compute_metrics_summary`

```python
def compute_metrics_summary(results: list[list[TokenMetrics]]) -> dict:
    """Returns aggregate stats from verification results."""
    # Returns: {
    #     "total_tokens": int,
    #     "exact_match_rate": float,
    #     "avg_prob": float,
    #     "avg_margin": float,           # Average of non-infinite margins
    #     "infinite_margin_rate": float, # Fraction of tokens with infinite margin
    #     "avg_logit_rank": float,
    #     "avg_gumbel_rank": float,
    # }
```

## Advanced Usage

### Temperature Zero Verification

If you don't know the exact sampling process of a provider, or if a provider doesn't support per-request sampling seeds, you can still verify outputs using temperature zero (greedy decoding). With `temperature=0`, the model deterministically selects the highest-probability token at each step, so the `top_k`, `top_p`, and `seed` parameters have no effect on the output.

In this case, pass arbitrary default values for the required parameters:

```python
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.0,  # Greedy decoding - deterministic
    top_k=50,         # Ignored when temperature=0
    top_p=0.95,       # Ignored when temperature=0
    seed=0,           # Ignored when temperature=0
)
```

### Using Quantization

```python
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=1.0, top_k=50, top_p=0.95, seed=42,
    vllm_kwargs={"quantization": "fp8"},
)
```

### Using FP8 KV Cache

```python
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=1.0, top_k=50, top_p=0.95, seed=42,
    vllm_kwargs={
        "kv_cache_dtype": "fp8",
        "calculate_kv_scales": True,
    },
)
```

### Custom Model Length

```python
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=1.0, top_k=50, top_p=0.95, seed=42,
    max_model_len=4096,
)
```

## Environment Variables

The package automatically sets `VLLM_USE_V1=1` (using `setdefault`, so you can override before importing). This enables vLLM v1 features required for prompt logprobs.

## Development

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

You can specify a different model for testing via environment variable:

```bash
TEST_MODEL=Qwen/Qwen3-1.7B pytest tests/ -v
```

## License

MIT
