# token-difr

Verify LLM outputs using Gumbel-Max sampling verification.

## Installation

```bash
pip install token-difr
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- vLLM >= 0.6.0
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
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    seed=42,
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
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    seed: int = 42,
    max_model_len: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    vllm_kwargs: dict[str, Any] | None = None,
    sampling_method: SamplingMethod = SamplingMethod.VLLM_GUMBEL_MAX,
    gpu_memory_utilization: float = 0.7,
    show_progress: bool = True,
) -> list[list[TokenMetrics]]:
```

**Parameters:**

- `outputs`: List of `TokenSequence` objects containing prompt and output token IDs
- `model_name`: HuggingFace model name (e.g., `"meta-llama/Llama-3.1-8B-Instruct"`)
- `temperature`: Sampling temperature used during generation (default: 1.0)
- `top_k`: Top-k sampling parameter (default: 50)
- `top_p`: Top-p (nucleus) sampling parameter (default: 0.95)
- `seed`: Random seed used during generation (default: 42)
- `max_model_len`: Maximum model context length. If None, auto-computed from outputs
- `dtype`: Model dtype, e.g., `torch.bfloat16`, `torch.float16` (default: `torch.bfloat16`)
- `vllm_kwargs`: Additional kwargs for vLLM's LLM constructor (e.g., for quantization)
- `sampling_method`: Sampling method to verify against (default: `VLLM_GUMBEL_MAX`)
- `gpu_memory_utilization`: Fraction of GPU memory to use (default: 0.7)
- `show_progress`: Whether to show progress bars (default: True)

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

## Advanced Usage

### Using Quantization

```python
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    vllm_kwargs={"quantization": "fp8"},
)
```

### Using FP8 KV Cache

```python
results = verify_outputs(
    outputs,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
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
    max_model_len=4096,
)
```

## Environment Variables

The package automatically sets `VLLM_USE_V1=1` (using `setdefault`, so you can override before importing). This enables vLLM v1 features required for prompt logprobs.

## License

MIT
