#%%
import argparse
import time
from statistics import mean

import torch
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer


def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def should_log_step(step_idx, total_steps, every):
    if step_idx <= 5:
        return True
    if step_idx == total_steps:
        return True
    return step_idx % every == 0


def print_step(prefix, step_idx, total_steps, input_len, step_ms, extra=""):
    print(
        f"{prefix} step {step_idx:>3}/{total_steps} | "
        f"model_input_len={input_len:>4} | "
        f"step_time={step_ms:>8.2f} ms{extra}"
    )


def print_checkpoint_table(prompt_len, times_no_cache, times_with_cache, memory_growth_mb):
    checkpoints = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100]
    checkpoints = [s for s in checkpoints if s <= len(times_no_cache)]

    print("\nCheckpoint Table (selected decode steps)")
    print(
        f"{'step':>4} | {'ctx_len':>7} | {'no_cache(ms)':>12} | "
        f"{'with_cache(ms)':>14} | {'speedup':>7} | {'cache_mb':>8}"
    )
    print("-" * 74)

    for step in checkpoints:
        i = step - 1
        ctx_len = prompt_len + i
        no_cache_ms = times_no_cache[i]
        with_cache_ms = times_with_cache[i]
        speedup = no_cache_ms / with_cache_ms if with_cache_ms > 0 else float("inf")
        cache_mb = memory_growth_mb[i]
        print(
            f"{step:>4} | {ctx_len:>7} | {no_cache_ms:>12.2f} | "
            f"{with_cache_ms:>14.2f} | {speedup:>7.2f} | {cache_mb:>8.2f}"
        )


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lecture demo: quantify KV-cache speed vs memory trade-off."
    )
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help="Hugging Face model id (default: gpt2).",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=positive_int,
        default=20,
        help="How many times to repeat the base prompt (default: 20).",
    )
    parser.add_argument(
        "--num-generated-tokens",
        type=positive_int,
        default=100,
        help="Number of decode steps to benchmark (default: 100).",
    )
    parser.add_argument(
        "--step-log-every",
        type=positive_int,
        default=1,
        help="Print step metrics every N steps (default: 1 = every step).",
    )
    parser.add_argument(
        "--preview-tokens",
        type=positive_int,
        default=40,
        help="How many generated tokens to decode for preview text (default: 40).",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=positive_int,
        default=16,
        help="Prompt tokens to use in warm-up pass (default: 16).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device selection (default: auto).",
    )
    return parser.parse_args()


def select_device(device_mode):
    if device_mode == "cpu":
        return torch.device("cpu")
    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but not available. Use --device auto or --device cpu."
            )
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
# 1. SETUP MODEL & TOKENIZER
args = parse_args()

MODEL_NAME = args.model_name
PROMPT_REPEAT = args.prompt_repeat
NUM_GENERATED_TOKENS = args.num_generated_tokens
STEP_LOG_EVERY = args.step_log_every
PREVIEW_TOKENS = args.preview_tokens
WARMUP_TOKENS = args.warmup_tokens

device = select_device(args.device)

print_section("0) Setup: model, prompt, and run configuration")
print(f"Model: {MODEL_NAME}")
print(f"Requested device mode: {args.device}")
print(f"Device: {device}")
print(f"Generated tokens per run: {NUM_GENERATED_TOKENS}")
print(f"Prompt repetition count: {PROMPT_REPEAT}")
print(f"Verbose step logging cadence: every {STEP_LOG_EVERY} steps")
print(f"Warm-up token count cap: {WARMUP_TOKENS}")
print(f"Preview text tokens: {PREVIEW_TOKENS}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Long initial context to make the KV cache difference immediate
prompt = "Deep learning architectures rely on self-attention mechanisms to sequence tokens. " * PROMPT_REPEAT
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
prompt_len = input_ids.shape[1]
batch_size = input_ids.shape[0]

cfg = model.config
n_layers = int(getattr(cfg, "n_layer", getattr(cfg, "num_hidden_layers", -1)))
n_heads = int(getattr(cfg, "n_head", getattr(cfg, "num_attention_heads", -1)))
hidden_size = int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", -1)))
head_dim = hidden_size // n_heads if n_heads > 0 and hidden_size > 0 else -1
dtype_bytes = next(model.parameters()).element_size()

print(f"Initial prompt length: {prompt_len} tokens")
print(f"Batch size: {batch_size}")
if n_layers > 0 and n_heads > 0 and hidden_size > 0:
    print(
        "Model internals used in KV estimate: "
        f"layers={n_layers}, heads={n_heads}, hidden={hidden_size}, head_dim={head_dim}"
    )
print(f"Model parameter dtype size: {dtype_bytes} bytes")

print("\nWarm-up pass to reduce one-time startup noise in timing...")
with torch.no_grad():
    warmup_len = min(prompt_len, WARMUP_TOKENS)
    _ = model(input_ids[:, :warmup_len], use_cache=True)
print("Warm-up complete.")

# Tracking metrics per step
times_no_cache = []
times_with_cache = []
generated_tokens_no_cache = []
generated_tokens_with_cache = []

# ==========================================
# PART 1: TIME BENCHMARKING
# ==========================================
print_section("1) Benchmark A: decoding WITHOUT KV cache")
print("Interpretation: each step recomputes the full prefix, so step time should grow with context length.")

# --- A. Without KV Cache ---
curr_input = input_ids.clone()
with torch.no_grad():
    for step in range(1, NUM_GENERATED_TOKENS + 1):
        input_len = curr_input.shape[1]
        t0 = time.perf_counter()
        outputs = model(curr_input, use_cache=False)
        t1 = time.perf_counter()

        step_ms = (t1 - t0) * 1000
        times_no_cache.append(step_ms)
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens_no_cache.append(int(next_token[0, 0].item()))
        curr_input = torch.cat([curr_input, next_token], dim=-1)

        if should_log_step(step, NUM_GENERATED_TOKENS, STEP_LOG_EVERY):
            print_step("[No cache ]", step, NUM_GENERATED_TOKENS, input_len, step_ms)

print_section("2) Benchmark B: decoding WITH KV cache")
print("Interpretation: step 1 is prefill (full prompt), later steps consume one token while reusing cached keys/values.")

# --- B. With KV Cache ---
curr_input = input_ids.clone()
past_key_values = None
with torch.no_grad():
    for step in range(1, NUM_GENERATED_TOKENS + 1):
        input_len = curr_input.shape[1]
        t0 = time.perf_counter()
        outputs = model(curr_input, past_key_values=past_key_values, use_cache=True)
        t1 = time.perf_counter()

        step_ms = (t1 - t0) * 1000
        times_with_cache.append(step_ms)
        past_key_values = outputs.past_key_values

        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens_with_cache.append(int(next_token[0, 0].item()))
        # Passing ONLY the single new token!
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

# Recompute memory from the same cache trajectory to populate exact values.
print_section("2.5) Memory pass: measuring the cache tax directly")
print("This pass isolates memory accounting so timing measurements stay clean.")
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

print_section("3) Quantitative summary: speedup purchased by memory")
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
if NUM_GENERATED_TOKENS > 1:
    per_token_cache_growth_mb = (memory_growth_mb[-1] - memory_growth_mb[0]) / (NUM_GENERATED_TOKENS - 1)
else:
    per_token_cache_growth_mb = 0.0

print(f"Total decoding time without cache: {total_no_cache_ms:,.2f} ms")
print(f"Total decoding time with cache:    {total_with_cache_ms:,.2f} ms")
print(f"Overall speedup from KV cache:     {overall_speedup:,.2f}x")
print()
print(f"Average step time without cache:   {avg_no_cache_ms:,.2f} ms")
print(f"Average step time with cache:      {avg_with_cache_ms:,.2f} ms")
print(f"Late-token speedup (last {late_window}): {late_speedup:,.2f}x")
print()
print(f"No-cache step-time growth (last/first):  {no_cache_growth:,.2f}x")
print(f"With-cache step-time growth (last/first): {with_cache_growth:,.2f}x")
print()
print(f"KV cache footprint after prefill:  {prefill_cache_mb:,.2f} MB")
print(f"KV cache footprint at end:         {peak_cache_mb:,.2f} MB")
print(f"Measured cache growth per token:   {per_token_cache_growth_mb:,.4f} MB/token")
if theoretical_mb_per_token is not None:
    print(f"Theoretical growth per token:      {theoretical_mb_per_token:,.4f} MB/token")

token_match = generated_tokens_no_cache == generated_tokens_with_cache
print()
print(f"Token-by-token outputs match between modes: {token_match}")
preview_tokens = generated_tokens_with_cache[: min(PREVIEW_TOKENS, len(generated_tokens_with_cache))]
preview_text = tokenizer.decode(preview_tokens).replace("\n", " ")
print(f"Generated continuation preview (first {len(preview_tokens)} tokens):")
print(preview_text)
print_checkpoint_table(prompt_len, times_no_cache, times_with_cache, memory_growth_mb)

print_section("4) Lecture takeaway")
print("KV cache converts an increasing-time decode process into a near-constant-time one.")
print("The price is a cache that grows approximately linearly with context length.")
print("This demo quantifies both sides of that trade-off on the same run.")

# ==========================================
# PART 3: LIVE VISUALIZATION
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("The KV Cache Trade-Off: Time vs. Memory", fontsize=14, fontweight='bold')

# Plot 1: Per-Token Execution Time
tokens_range = list(range(1, NUM_GENERATED_TOKENS + 1))
ax1.plot(tokens_range, times_no_cache, label="Without KV Cache (Recompute)", color='#d9534f', linewidth=2.5)
ax1.plot(tokens_range, times_with_cache, label="With KV Cache (O(1) Step Time)", color='#5cb85c', linewidth=2.5)
ax1.set_title("1. Execution Time per Generated Token")
ax1.set_xlabel("Generated Token Index")
ax1.set_ylabel("Step Time (ms)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()
ax1.text(
    0.02,
    0.98,
    (
        f"Total speedup: {overall_speedup:.2f}x\n"
        f"Late-token speedup: {late_speedup:.2f}x\n"
        f"No-cache growth: {no_cache_growth:.2f}x"
    ),
    transform=ax1.transAxes,
    verticalalignment='top',
    fontsize=9,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
)

# Plot 2: KV Cache Memory Growth
ax2.plot(tokens_range, memory_growth_mb, color='#0275d8', linewidth=2.5)
ax2.fill_between(tokens_range, memory_growth_mb, color='#0275d8', alpha=0.15)
ax2.set_title("2. KV Cache Footprint (The Memory Tax)")
ax2.set_xlabel("Generated Token Index")
ax2.set_ylabel("VRAM Used by Cache (MB)")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.text(
    0.02,
    0.98,
    (
        f"Prefill cache: {prefill_cache_mb:.2f} MB\n"
        f"Peak cache: {peak_cache_mb:.2f} MB\n"
        f"Measured growth: {per_token_cache_growth_mb:.4f} MB/token"
    ),
    transform=ax2.transAxes,
    verticalalignment='top',
    fontsize=9,
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
)

plt.tight_layout(rect=[0, 0.02, 1, 0.95])
plt.show()
# %%
