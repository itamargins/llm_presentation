#%%
import time
from dataclasses import dataclass
from statistics import mean

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utility functions & color loggers
from demo_utils import (
    c_green,
    c_val,
    c_yellow,
    select_device,
    sync_device,
    print_section,
    print_step,
    should_log_step,
    get_kv_cache_memory_mb,
    apply_safety_cap,
    print_descriptive_results,
    print_lecture_takeaway,
)


# ==============================================================================
# LECTURE DEMO CONFIGURATION
# Modify model choice and prominent parameters easily right here!
# ==============================================================================
@dataclass
class DemoConfig:
    # Model options: "gpt2", "gpt2-medium", "Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B"
    model_name: str = "gpt2"
    base_sentence: str = "Deep learning architectures rely on self-attention mechanisms to sequence tokens. "
    # Prompt repeat count (will be automatically capped if it exceeds model max context)
    prompt_repeat: int = 1200
    
    # Number of decode steps to benchmark
    num_generated_tokens: int = 100
    
    # Execution & logging settings
    device_mode: str = "auto"  # Options: "auto", "cuda", "cpu"
    step_log_every: int = 10


#%%
# 1. SETUP MODEL, TOKENIZER & SAFETY CAPS
cfg = DemoConfig()

MODEL_NAME = cfg.model_name
PROMPT_REPEAT = cfg.prompt_repeat
NUM_GENERATED_TOKENS = cfg.num_generated_tokens
STEP_LOG_EVERY = cfg.step_log_every

device = select_device(cfg.device_mode)

print_section("0) Setup: model, prompt, and run configuration")
print(f"Model: {c_val(MODEL_NAME)}")
print(f"Device: {c_val(str(device))}")
print(f"Generated tokens requested: {c_val(str(NUM_GENERATED_TOKENS))}")
print(f"Initial prompt repetition target: {c_val(str(PROMPT_REPEAT))}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Construct base prompt
base_sentence = cfg.base_sentence
prompt = base_sentence * PROMPT_REPEAT
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Enforce dynamic safety cap to prevent context overflow crashes
input_ids, max_model_ctx = apply_safety_cap(model, input_ids, NUM_GENERATED_TOKENS)

prompt_len = input_ids.shape[1]
batch_size = input_ids.shape[0]

# Model structural parameters for theoretical memory calculation
n_layers = int(getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", -1)))
n_heads = int(getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", -1)))
hidden_size = int(getattr(model.config, "n_embd", getattr(model.config, "hidden_size", -1)))
dtype_bytes = next(model.parameters()).element_size()

print(f"Final safe prompt length: {c_green(str(prompt_len))} tokens (Max model window: {max_model_ctx})")
print(f"Batch size: {batch_size}")

print(f"\n{c_yellow('Performing CUDA warm-up pass over full context length...')}")
with torch.no_grad():
    _ = model(input_ids, use_cache=False)
    _ = model(input_ids, use_cache=True)
sync_device(device)
print(c_green("Warm-up complete."))

times_no_cache = []
times_with_cache = []
generated_tokens_no_cache = []
generated_tokens_with_cache = []

# ==============================================================================
# PART 1: TIME BENCHMARKING
# ==============================================================================
print_section("1) Benchmark A: decoding WITHOUT KV cache")

curr_input = input_ids.clone()
with torch.no_grad():
    for step in range(1, NUM_GENERATED_TOKENS + 1):
        input_len = curr_input.shape[1]
        
        sync_device(device)
        t0 = time.perf_counter()
        outputs = model(curr_input, use_cache=False)
        sync_device(device)
        t1 = time.perf_counter()

        step_ms = (t1 - t0) * 1000
        times_no_cache.append(step_ms)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens_no_cache.append(int(next_token[0, 0].item()))
        curr_input = torch.cat([curr_input, next_token], dim=-1)

        if should_log_step(step, NUM_GENERATED_TOKENS, STEP_LOG_EVERY):
            print_step("[No cache ]", step, NUM_GENERATED_TOKENS, input_len, step_ms)

print_section("2) Benchmark B: decoding WITH KV cache")

curr_input = input_ids.clone()
past_key_values = None
with torch.no_grad():
    for step in range(1, NUM_GENERATED_TOKENS + 1):
        input_len = curr_input.shape[1]

        sync_device(device)
        t0 = time.perf_counter()
        outputs = model(curr_input, past_key_values=past_key_values, use_cache=True)
        sync_device(device)
        t1 = time.perf_counter()

        step_ms = (t1 - t0) * 1000
        times_with_cache.append(step_ms)
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens_with_cache.append(int(next_token[0, 0].item()))
        curr_input = next_token

        if should_log_step(step, NUM_GENERATED_TOKENS, STEP_LOG_EVERY):
            cache_tokens = prompt_len + (step - 1)
            print_step(
                "[With cache]",
                step,
                NUM_GENERATED_TOKENS,
                input_len,
                step_ms,
                extra=f" | cached_tokens~{cache_tokens:>4}",
            )

# ==============================================================================
# PART 2: MEMORY BENCHMARKING
# ==============================================================================
print_section("2.5) Memory pass: measuring cache footprint")
past_key_values = None
curr_input = input_ids.clone()
memory_growth_mb = []
with torch.no_grad():
    for _ in range(NUM_GENERATED_TOKENS):
        outputs = model(curr_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        memory_growth_mb.append(get_kv_cache_memory_mb(past_key_values))
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        curr_input = next_token

theoretical_bytes_per_token = None
if n_layers > 0 and hidden_size > 0:
    theoretical_bytes_per_token = 2 * batch_size * n_layers * hidden_size * dtype_bytes
theoretical_mb_per_token = (
    theoretical_bytes_per_token / (1024 ** 2)
    if theoretical_bytes_per_token is not None
    else None
)

# Compute quantitative summary values
total_no_cache_ms = sum(times_no_cache)
total_with_cache_ms = sum(times_with_cache)
avg_no_cache_ms = mean(times_no_cache)
avg_with_cache_ms = mean(times_with_cache)
overall_speedup = total_no_cache_ms / total_with_cache_ms if total_with_cache_ms > 0 else float("inf")

late_window = min(20, NUM_GENERATED_TOKENS)
late_no_cache_ms = mean(times_no_cache[-late_window:])
late_with_cache_ms = mean(times_with_cache[-late_window:])
late_speedup = late_no_cache_ms / late_with_cache_ms if late_with_cache_ms > 0 else float("inf")

no_cache_growth = times_no_cache[-1] / times_no_cache[0] if times_no_cache[0] > 0 else float("inf")
with_cache_growth = times_with_cache[-1] / times_with_cache[0] if times_with_cache[0] > 0 else float("inf")

peak_cache_mb = memory_growth_mb[-1]
prefill_cache_mb = memory_growth_mb[0]
per_token_cache_growth_mb = (
    (memory_growth_mb[-1] - memory_growth_mb[0]) / (NUM_GENERATED_TOKENS - 1)
    if NUM_GENERATED_TOKENS > 1
    else 0.0
)
token_match = generated_tokens_no_cache == generated_tokens_with_cache

# Output detailed results with descriptions
print_descriptive_results(
    total_no_cache_ms,
    total_with_cache_ms,
    overall_speedup,
    avg_no_cache_ms,
    avg_with_cache_ms,
    late_window,
    late_speedup,
    no_cache_growth,
    with_cache_growth,
    prefill_cache_mb,
    peak_cache_mb,
    per_token_cache_growth_mb,
    theoretical_mb_per_token,
    token_match,
    prompt_len,
    times_no_cache,
    times_with_cache,
    memory_growth_mb,
)

print_lecture_takeaway(overall_speedup, late_speedup, per_token_cache_growth_mb)
# ==============================================================================
# PART 3: LIVE VISUALIZATION
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.2))
fig.suptitle(
    f"The KV Cache Trade-Off ({MODEL_NAME}): Time vs. Memory",
    fontsize=14,
    fontweight="bold",
)

tokens_range = list(range(1, NUM_GENERATED_TOKENS + 1))
ax1.plot(
    tokens_range,
    times_no_cache,
    label="Without KV Cache (Recompute)",
    color="#d9534f",
    linewidth=2.5,
)
ax1.plot(
    tokens_range,
    times_with_cache,
    label="With KV Cache (O(1) Step Time)",
    color="#5cb85c",
    linewidth=2.5,
)
ax1.set_title("1. Execution Time per Generated Token")
ax1.set_xlabel("Generated Token Index")
ax1.set_ylabel("Step Time (ms)")
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(loc="upper right")

# Add summary box on Plot 1
ax1.text(
    0.02,
    0.98,
    (
        f"Total speedup: {overall_speedup:.2f}x\n"
        f"Late-token speedup: {late_speedup:.2f}x\n"
        f"No-cache growth: {no_cache_growth:.2f}x"
    ),
    transform=ax1.transAxes,
    verticalalignment="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)

# Annotate Token 2 Spike if present
if len(times_with_cache) >= 2:
    token2_time = times_with_cache[1]
    ax1.annotate(
        "Token 2 Spike:\nGPU Dynamic Tensor\nReallocation (torch.cat)",
        xy=(2, token2_time),
        xytext=(min(15, NUM_GENERATED_TOKENS // 3), max(times_no_cache) * 0.45),
        arrowprops=dict(facecolor="#333333", shrink=0.08, width=1.5, headwidth=6),
        fontsize=8.5,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd", edgecolor="#ffeeba"),
    )

# Plot 2: Memory Footprint
ax2.plot(tokens_range, memory_growth_mb, color="#0275d8", linewidth=2.5)
ax2.fill_between(tokens_range, memory_growth_mb, color="#0275d8", alpha=0.15)
ax2.set_title("2. KV Cache Footprint (The Memory Tax)")
ax2.set_xlabel("Generated Token Index")
ax2.set_ylabel("VRAM Used by Cache (MB)")
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.text(
    0.02,
    0.98,
    (
        f"Prefill cache: {prefill_cache_mb:.2f} MB\n"
        f"Peak cache: {peak_cache_mb:.2f} MB\n"
        f"Measured growth: {per_token_cache_growth_mb:.4f} MB/token"
    ),
    transform=ax2.transAxes,
    verticalalignment="top",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)

# Footnote Explanation under the plots
explanation_text = (
    "Note on the Step 2 Spike: The initial jump in 'With Cache' step time at Token 2 is caused by PyTorch GPU VRAM reallocation.\n"
    "Appending the first generated token requires allocating a new contiguous buffer and executing `torch.cat()` across all layers.\n"
    "From Step 3 onwards, CUDA memory pools stabilize, demonstrating true O(1) decode latency. (Production engines avoid this via PagedAttention)."
)
fig.text(
    0.5,
    0.015,
    explanation_text,
    ha="center",
    va="bottom",
    fontsize=8.5,
    style="italic",
    bbox=dict(boxstyle="square,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", lw=0.8),
)

plt.tight_layout(rect=[0, 0.07, 1, 0.95])
plt.show()
plt.savefig(f"./kv_cache_benchmark_{MODEL_NAME}.png", dpi=300)
# %%