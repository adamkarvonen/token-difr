# %%

import os
import tinker
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TINKER_API_KEY")

# %%

service_client = tinker.ServiceClient(api_key=api_key)
sampling_client = service_client.create_sampling_client(base_model="meta-llama/Llama-3.1-8B-Instruct")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

# %%

# Format prompt using Llama 3 chat template
messages = [{"role": "user", "content": "How should I boil an egg?"}]

# Use HuggingFace's built-in chat template
prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
prompt_tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
prompt = tinker.ModelInput.from_ints(prompt_tokens)

# Step 1: Generate the completion
params = tinker.SamplingParams(max_tokens=50, temperature=1.0, top_k=50, seed=42)
result = sampling_client.sample(
    prompt=prompt,
    sampling_params=params,
    num_samples=10,
).result()


generated_tokens = result.sequences[0].tokens
print("Generated text:")
print(tokenizer.decode(generated_tokens))
# %%

print(len(result.sequences))

for sequence in result.sequences:
    print(f"\n\n\nGenerated text: {tokenizer.decode(sequence.tokens)}")


unique_sequences = set(tokenizer.decode(sequence.tokens) for sequence in result.sequences)
print(f"Number of unique sequences: {len(unique_sequences)}")
# %%

# Step 2: Concatenate prompt + generated tokens and get top-k logprobs
full_sequence = prompt_tokens + generated_tokens
full_prompt = tinker.ModelInput.from_ints(full_sequence)

logprob_result = sampling_client.sample(
    prompt=full_prompt,
    sampling_params=tinker.SamplingParams(max_tokens=1),  # Just need prefill
    num_samples=1,
    include_prompt_logprobs=True,
    topk_prompt_logprobs=20,
).result()

# topk_prompt_logprobs is a list of (token_id, logprob) tuples for each position
print("\nTop-k logprobs for each token:")
for i, topk in enumerate(logprob_result.topk_prompt_logprobs):
    token_str = repr(tokenizer.decode([full_sequence[i]]))
    if topk is None:
        print(f"  {i}: {token_str:20} -> None (first token)")
    else:
        # Show top 5 for display, but all 50 are available
        top5 = topk[:5]
        top5_strs = [f"{tokenizer.decode([t])!r}: {lp:.2f}" for t, lp in top5]
        print(f"  {i}: actual={token_str:15} top5=[{', '.join(top5_strs)}]")
# %%
