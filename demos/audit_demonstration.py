import json
from dataclasses import asdict

from dotenv import load_dotenv

load_dotenv()

from token_difr import audit_provider, construct_prompts

HF_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
PROVIDERS = [
    # "parasail/fp8",
    # "wandb/bf16",
    # "deepinfra/fp8",
    # "novita/fp8",
    # "siliconflow/fp8",
    # "together",
    # "fireworks/fp8",
    # "cerebras",
    "alibaba"
]

# Audit parameters
N_PROMPTS = 10
MAX_TOKENS = 200
SEED = 42
TOP_K = 50
TOP_P = 0.95
TEMPERATURE = 0.0


def save_results(results: dict, output_file: str) -> None:
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


def main():
    prompts = construct_prompts(
        n_prompts=N_PROMPTS,
        model_name=HF_MODEL,
        system_prompt="You are a helpful assistant.",
    )
    print(f"Constructed {len(prompts)} prompts")

    # Initialize results structure with metadata
    results = {
        "model": HF_MODEL,
        "parameters": {
            "n_prompts": N_PROMPTS,
            "max_tokens": MAX_TOKENS,
            "seed": SEED,
            "top_k": TOP_K,
            "top_p": TOP_P,
            "temperature": TEMPERATURE,
        },
        "providers": {},
    }

    safe_model_name = HF_MODEL.replace("/", "_")
    output_file = f"{safe_model_name}_audit_results.json"

    # Write initial file so we can watch progress
    save_results(results, output_file)
    print(f"Results will be saved to {output_file}")

    for provider in PROVIDERS:
        print(f"\nAuditing provider: {provider}")
        try:
            result = audit_provider(
                prompts,
                model=HF_MODEL,
                provider=provider,
                max_tokens=MAX_TOKENS,
                seed=SEED,
                top_k=TOP_K,
                top_p=TOP_P,
                temperature=TEMPERATURE,
            )

            results["providers"][provider] = asdict(result)

            print(f"  Total tokens: {result.total_tokens}")
            print(f"  Exact match rate: {result.exact_match_rate:.2%}")
            print(f"  Avg probability: {result.avg_prob:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results["providers"][provider] = {"error": str(e)}

        # Save after each provider completes
        save_results(results, output_file)

    print(f"\nAll results saved to {output_file}")


if __name__ == "__main__":
    main()
