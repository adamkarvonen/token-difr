# token-difr

Verify that LLM API providers are running the models they claim.

Generate tokens from any provider via OpenRouter, then verify against a reference provider that returns logprobs for prompt tokens. By default we use Fireworks as the reference because they support most open-weight models. You can also verify against locally hosted models (via vLLM) or Tinker.

Each token is an independent verification point—100 prompts × 200 tokens = 20,000 data points with minimal API cost.

## Why this approach?

Traditional model evaluations require models to answer questions, often generating 1000+ tokens per question with high variance. Token-level verification treats each generated token as a single forward pass, providing high statistical significance with fewer tokens and lower cost.

## Installation

```bash
pip install token-difr
```

## Requirements

- Python >= 3.10
- OpenRouter API key (for generation)
- Fireworks API key (for verification)

Set your API keys:
```bash
export OPENROUTER_API_KEY="your-key"
export FIREWORKS_API_KEY="your-key"
```

## Quick Start

```python
from token_difr import audit_provider, construct_prompts

# Load prompts from WildChat dataset
prompts = construct_prompts(
    n_prompts=100,
    model_name="meta-llama/Llama-3.3-70B-Instruct",
    system_prompt="You are a helpful assistant.",
)

# Audit a provider
result = audit_provider(
    prompts,
    model="meta-llama/Llama-3.3-70B-Instruct",
    provider="together",  # OpenRouter provider to test
    max_tokens=200,
)

print(result)
# example AuditResult(98.3% match rate, 18421 tokens across 100 sequences)
```

## How It Works

1. **Generate**: Send prompts to a provider via OpenRouter and collect responses
2. **Tokenize**: Convert responses to token IDs using the model's HuggingFace tokenizer
3. **Verify**: Send token sequences to Fireworks to get logprobs for each position
4. **Compare**: Check if the generated tokens match what Fireworks would have produced

All verification uses `temperature=0` (greedy decoding) because sampling seeds are not standardized across providers. With greedy decoding, the highest-probability token is always selected, making outputs deterministic and comparable.

## Understanding Results

```python
@dataclass
class AuditResult:
    exact_match_rate: float  # Primary metric: fraction of tokens that match
    avg_prob: float          # Average probability of generated tokens
    avg_margin: float        # Average log-prob difference from top token
    total_tokens: int        # Total tokens verified
    n_sequences: int         # Number of sequences verified
```

## Interpreting Results

### What Scores Mean (and Don't Mean)

The exact match rate is the primary metric because it's most interpretable: it measures what fraction of generated tokens exactly match what the reference provider would produce.

**Important**: Low match rate scores indicate *divergence from reference*, not necessarily low quality. A high exact match rate (>98%) with a trusted reference can give high confidence in the provided model, but several factors can cause a low exact match rate.

Divergence can occur for several reasons:

| Cause | Typical Impact | How to Identify |
|-------|---------------|-----------------|
| Different system prompt | 5-20% drop | Consistent across all prompts |
| Different tokenization format | 1-5% drop | Often affects prompt boundaries |
| Quantization | 1-3% drop | Consistent small reduction |
| Tokenization drift from re-encoding | 1-3% drop | Random distribution of mismatches |
| Genuinely different model | 20%+ drop | Often correlates with semantic differences |

### Setting Thresholds

There's no universal threshold that separates "good" from "bad" providers. Instead:

- **Baseline first**: Run the same model on multiple providers to establish what scores to expect
- **Compare relatively**: A provider scoring 94% when others score 98% warrants investigation

## Model Registry

Fireworks uses different model names than HuggingFace, so you must register a Fireworks model name before auditing. The package includes common models:

```python
from token_difr import FIREWORKS_MODEL_REGISTRY

# Built-in models
print(FIREWORKS_MODEL_REGISTRY.keys())
# dict_keys(['meta-llama/Llama-3.3-70B-Instruct', 'meta-llama/Llama-3.1-8B-Instruct', ...])
```

In addition, OpenRouter sometimes uses different model names, but often just uses the lower case version of the HuggingFace name.

### Adding a New Model

```python
from token_difr import register_fireworks_model, register_openrouter_model

# Required: map HuggingFace name to Fireworks name
register_fireworks_model(
    "mistralai/Mistral-Large-2",
    "accounts/fireworks/models/mistral-large-2"
)

# Occasionally OpenRouter names differ from hf_name.lower() as well
register_openrouter_model(
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "qwen/qwen3-235b-a22b-2507"
)
```

Check each provider's documentation for exact model names.

## Auditing Multiple Providers

```python
import json
from dataclasses import asdict
from token_difr import audit_provider, construct_prompts

MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
PROVIDERS = ["together", "fireworks/fp8", "deepinfra/fp8", "novita/fp8"]

prompts = construct_prompts(n_prompts=100, model_name=MODEL)
results = {}

for provider in PROVIDERS:
    result = audit_provider(prompts, model=MODEL, provider=provider, max_tokens=200)
    results[provider] = asdict(result)
    print(f"{provider}: {result.exact_match_rate:.1%} match rate")

# Save results
with open("audit_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## Why Fireworks as the Reference?

Fireworks is used as the default verification backend because:
1. **Prompt logprobs**: Most providers only return logprobs for generated tokens. Fireworks returns logprobs for prompt tokens too, enabling verification.
2. **Model coverage**: Fireworks hosts most popular open-weight models.
3. **API-only**: No local GPU required.

### Trusting the Reference

Since we're verifying providers against Fireworks, we need confidence that Fireworks itself is correct. Two approaches:

**Option 1: Cross-check with first-party APIs**

Some model creators host their own APIs. For example, Moonshot hosts Kimi K2. Verify Fireworks gets high match rates against the first-party source:

```python
# Verify Fireworks matches Moonshot's own Kimi K2 endpoint
result = audit_provider(
    prompts,
    model="moonshotai/Kimi-K2-Thinking",
    provider="moonshotai",  # First-party provider
)
# If this passes, Fireworks is trustworthy for this model
```

**Option 2: One-time local verification**

Generate reference outputs from a locally-hosted model, then save the token sequences. You can use these saved tokens to periodically verify that Fireworks (or any other provider) continues to match:

```python
from token_difr import verify_outputs, TokenSequence
import json

# One-time: generate and save reference tokens from local model
# ... generate sequences locally with vLLM ...
# ... save to reference_tokens.json ...

# Periodic verification: load saved tokens and verify provider
with open("reference_tokens.json") as f:
    data = json.load(f)
    sequences = [TokenSequence(**s) for s in data]

results = verify_outputs(
    sequences,
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    temperature=0.0,
    top_k=50,
    top_p=0.95,
    seed=42,
)
```

## Advanced: Manual Verification Workflow

For more control, you can run the three steps separately:

```python
import asyncio
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from token_difr import (
    verify_outputs_fireworks,
    compute_metrics_summary,
    FIREWORKS_MODEL_REGISTRY,
    get_openrouter_name,
)
from token_difr.openrouter_api import (
    generate_openrouter_responses,
    tokenize_openrouter_responses,
)

async def manual_audit():
    model = "meta-llama/Llama-3.1-8B-Instruct"

    # Setup
    openrouter = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="your-openrouter-key",
    )
    fireworks = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="your-fireworks-key",
    )
    tokenizer = AutoTokenizer.from_pretrained(model)

    conversations = [
        [{"role": "user", "content": "What is the capital of France?"}],
        [{"role": "user", "content": "Explain photosynthesis briefly."}],
    ]

    # Step 1: Generate from OpenRouter
    responses = await generate_openrouter_responses(
        client=openrouter,
        conversations=conversations,
        model=get_openrouter_name(model),
        provider="together",
        temperature=0.0,
        max_tokens=100,
        seed=42,
    )

    # Step 2: Tokenize responses
    sequences = tokenize_openrouter_responses(
        conversations, responses, tokenizer, max_tokens=100
    )

    # Step 3: Verify against Fireworks
    results = await verify_outputs_fireworks(
        sequences,
        vocab_size=len(tokenizer),
        temperature=0.0,
        top_k=50,
        top_p=0.95,
        seed=42,
        client=fireworks,
        model=FIREWORKS_MODEL_REGISTRY[model],
    )

    summary = compute_metrics_summary(results)
    print(f"Exact match rate: {summary['exact_match_rate']:.1%}")

asyncio.run(manual_audit())
```

## Advanced: Local Model Verification

For full control and to verify without trusting any API, use local vLLM-based verification:

```python
from token_difr import verify_outputs, TokenSequence

# Token sequences from an untrusted source
sequences = [
    TokenSequence(
        prompt_token_ids=[128000, 2323, 374, 264, 1296],
        output_token_ids=[264, 1296, 13, 578, 4320],
    )
]

# Verify against local model
results = verify_outputs(
    sequences,
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    temperature=0.0,
    top_k=50,
    top_p=0.95,
    seed=42,
)
```

This requires a CUDA-capable GPU and vLLM.

## Use Case: System Prompt Detection

Detect if a provider modified the system prompt by verifying with your expected prompt:

```python
# Generate with unknown system prompt
# Verify assuming "You are a helpful assistant."
# Low match rate suggests the actual system prompt differs
```

See `system_prompt_detection.py` for a full example.

## Limitations

- **Temperature must be 0**: Sampling seeds are not standardized across providers, so only greedy decoding produces comparable outputs.
- **Tokenization edge cases**: `encode(decode(tokens))` may not equal original tokens, causing ~1-3% of mismatches even for identical models.
- **Model availability**: Both OpenRouter and Fireworks must support the model.
- **Quantization effects**: FP8 vs BF16 quantization can cause small differences in outputs.

## API Reference

### Core Functions

- `audit_provider(conversations, model, provider, ...)` - High-level audit function
- `construct_prompts(n_prompts, model_name, ...)` - Load prompts from WildChat dataset
- `verify_outputs(sequences, model_name, ...)` - Local vLLM verification
- `verify_outputs_fireworks(sequences, ...)` - Fireworks API verification
- `verify_outputs_tinker(sequences, ...)` - Tinker API verification

### Model Registry

- `FIREWORKS_MODEL_REGISTRY` - Dict mapping HuggingFace names to Fireworks names
- `OPENROUTER_MODEL_REGISTRY` - Dict for non-standard OpenRouter names
- `register_fireworks_model(hf_name, fireworks_name)` - Add a Fireworks mapping
- `register_openrouter_model(hf_name, openrouter_name)` - Add an OpenRouter mapping
- `get_openrouter_name(hf_name)` - Get OpenRouter name (uses registry or lowercase fallback)

### Data Classes

- `TokenSequence(prompt_token_ids, output_token_ids)` - Input for verification
- `TokenMetrics(exact_match, prob, margin, logit_rank, gumbel_rank)` - Per-token results
- `AuditResult(exact_match_rate, avg_prob, avg_margin, ...)` - Aggregate audit results

## License

MIT
