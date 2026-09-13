#%%
import time
import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

#%%
# 1. SETUP MODEL & TOKENIZER
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Long initial context to make the KV cache difference immediate
prompt = "Deep learning architectures rely on self-attention mechanisms to sequence tokens. " * 20
input_ids = tokenizer.encode(prompt, return_tensors="pt")
num_generated_tokens = 100

print(f"Initial Prompt Length: {input_ids.shape[1]} tokens")
print(f"Generating {num_generated_tokens} new tokens...\n")

# Tracking metrics per step
times_no_cache = []
times_with_cache = []

# ==========================================
# PART 1: TIME BENCHMARKING
# ==========================================

# --- A. Without KV Cache ---
curr_input = input_ids.clone()
with torch.no_grad():
    for _ in range(num_generated_tokens):
        t0 = time.perf_counter()
        outputs = model(curr_input, use_cache=False)
        t1 = time.perf_counter()
        
        times_no_cache.append((t1 - t0) * 1000) # Convert to ms
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        curr_input = torch.cat([curr_input, next_token], dim=-1)

# --- B. With KV Cache ---
curr_input = input_ids.clone()
past_key_values = None
with torch.no_grad():
    for _ in range(num_generated_tokens):
        t0 = time.perf_counter()
        outputs = model(curr_input, past_key_values=past_key_values, use_cache=True)
        t1 = time.perf_counter()
        
        times_with_cache.append((t1 - t0) * 1000) # Convert to ms
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        # Passing ONLY the single new token!
        curr_input = next_token 


#%%
# ==========================================
# PART 2: MEMORY CALCULATION (THE TAX)
# ==========================================
def get_kv_cache_memory_mb(past_key_values):
    """
    Computes exact VRAM footprint of HuggingFace past_key_values in MB.
    Works across older tuples, modern DynamicCache, and nested cache objects.
    """
    if past_key_values is None:
        return 0.0

    total_bytes = 0

    # 1. Modern HuggingFace Cache objects (DynamicCache, QuantizedCache, etc.)
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        # Check if caches contain layer tensors directly
        if len(past_key_values.key_cache) > 0 and isinstance(past_key_values.key_cache[0], torch.Tensor):
            for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
                total_bytes += k_tensor.nelement() * k_tensor.element_size()
                total_bytes += v_tensor.nelement() * v_tensor.element_size()
            return total_bytes / (1024 ** 2)

    # 2. Modern Cache object structured by layers (e.g. past_key_values.layers or tuple of layer objects)
    if hasattr(past_key_values, "__len__"):
        for layer in past_key_values:
            # Tuple/List per layer: (keys, values)
            if isinstance(layer, (tuple, list)):
                for item in layer:
                    if isinstance(item, torch.Tensor):
                        total_bytes += item.nelement() * item.element_size()
            # Object per layer with keys/values attributes
            elif hasattr(layer, "keys") and hasattr(layer, "values"):
                if isinstance(layer.keys, torch.Tensor):
                    total_bytes += layer.keys.nelement() * layer.keys.element_size()
                if isinstance(layer.values, torch.Tensor):
                    total_bytes += layer.values.nelement() * layer.values.element_size()

    # 3. Recursive fallback: find all PyTorch tensors inside past_key_values
    if total_bytes == 0:
        def extract_tensors(obj):
            tensors = []
            if isinstance(obj, torch.Tensor):
                tensors.append(obj)
            elif isinstance(obj, (tuple, list)):
                for item in obj:
                    tensors.extend(extract_tensors(item))
            elif hasattr(obj, "__dict__"):
                for val in obj.__dict__.values():
                    tensors.extend(extract_tensors(val))
            return tensors

        all_tensors = extract_tensors(past_key_values)
        for t in all_tensors:
            total_bytes += t.nelement() * t.element_size()

    return total_bytes / (1024 ** 2)

memory_growth_mb = []
past_key_values = None
curr_input = input_ids.clone()

# Track memory footprint as context length grows
with torch.no_grad():
    for step in range(num_generated_tokens):
        outputs = model(curr_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        mem_mb = get_kv_cache_memory_mb(past_key_values)
        memory_growth_mb.append(mem_mb)
        
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        curr_input = next_token

# ==========================================
# PART 3: LIVE VISUALIZATION
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("The KV Cache Trade-Off: Time vs. Memory", fontsize=14, fontweight='bold')

# Plot 1: Per-Token Execution Time
tokens_range = list(range(1, num_generated_tokens + 1))
ax1.plot(tokens_range, times_no_cache, label="Without KV Cache (Recompute)", color='#d9534f', linewidth=2.5)
ax1.plot(tokens_range, times_with_cache, label="With KV Cache (O(1) Step Time)", color='#5cb85c', linewidth=2.5)
ax1.set_title("1. Execution Time per Generated Token")
ax1.set_xlabel("Generated Token Index")
ax1.set_ylabel("Step Time (ms)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot 2: KV Cache Memory Growth
ax2.plot(tokens_range, memory_growth_mb, color='#0275d8', linewidth=2.5)
ax2.fill_between(tokens_range, memory_growth_mb, color='#0275d8', alpha=0.15)
ax2.set_title("2. KV Cache Footprint (The Memory Tax)")
ax2.set_xlabel("Generated Token Index")
ax2.set_ylabel("VRAM Used by Cache (MB)")
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
# %%
