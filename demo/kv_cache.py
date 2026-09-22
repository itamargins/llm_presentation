#%%
import textwrap
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
    c_dim,
    c_bold,
    select_device,
    sync_device,
    print_section,
    print_step,
    should_log_step,
    get_kv_cache_memory_mb,
    apply_safety_cap,
    prompt_token_budget,
    print_descriptive_results,
    print_lecture_takeaway,
    print_try_this_next,
    print_narration,
    compute_attention_architecture,
    print_attention_architecture_explanation,
    print_prefill_vs_decode_explanation,
    get_quantized_cache_memory_mb,
    print_quantized_cache_comparison,
)


# ==============================================================================
# >>>>>>>>>>>>>>>>>>>>>>>>  FRONT-END: EDIT THIS BLOCK  <<<<<<<<<<<<<<<<<<<<<<<<
#
# Everything a presenter normally touches lives here. Sections A-B are the
# interactive knobs (what you ask the model, how much it writes); C-E are the
# benchmark/plumbing knobs.
# ==============================================================================
@dataclass
class DemoConfig:
    # --- A. THE PROMPT: type any free text here --------------------------------
    # The model continues this text; the continuation is printed in Step 9.
    user_prompt: str = (
        "The key advantage of caching keys and values during inference is"
    )
    # True  -> the demo asks you to type a prompt in the terminal on startup
    #          (pressing Enter alone keeps user_prompt above).
    # False -> use user_prompt above without asking.
    ask_for_prompt_at_runtime: bool = False

    # --- B. HOW MUCH TO GENERATE ----------------------------------------------
    # Number of decode steps to benchmark (= length of the printed continuation)
    num_generated_tokens: int = 100

    # --- C. WHICH MODEL -------------------------------------------------------
    # Options: "gpt2", "gpt2-medium", "Qwen/Qwen2.5-0.5B", "meta-llama/Llama-3.2-1B"
    model_name: str = "gpt2"

    # --- D. SYNTHETIC LONG-CONTEXT PADDING (benchmark knob) -------------------
    # A free-text prompt is only a handful of tokens -- far too short to expose
    # the quadratic cost of decoding without a cache. This filler sentence is
    # therefore PREPENDED to user_prompt to fill the context window. The user
    # prompt always stays last, so generation still continues YOUR text.
    # Set context_filler_repeat = 0 for a pure short-prompt interactive run.
    # (Repeat count is automatically capped to the model's max context.)
    context_filler_sentence: str = "Deep learning architectures rely on self-attention mechanisms to sequence tokens. "
    context_filler_repeat: int = 1200

    # --- E. EXECUTION & LOGGING -----------------------------------------------
    device_mode: str = "auto"  # Options: "auto", "cuda", "cpu"
    step_log_every: int = 10

    # BONUS (optional, may not be used): quantized KV cache memory comparison.
    # Requires an optional dependency ("pip install optimum-quanto" or "pip install hqq"),
    # and is skipped cleanly with a message if it isn't installed.
    run_quantized_cache_bonus: bool = False
    quantized_cache_backend: str = "quanto"  # Options: "quanto", "hqq"
    quantized_cache_nbits: int = 4


# ==============================================================================
# PROMPT / TEXT HELPERS
# ==============================================================================
def resolve_user_prompt(cfg):
    """Return the text the model will continue, asked interactively if enabled."""
    if not cfg.ask_for_prompt_at_runtime:
        return cfg.user_prompt

    try:
        typed = input(c_yellow("\nEnter a prompt (Enter alone = use the default): ")).strip()
    except EOFError:
        # No interactive stdin (piped run, some notebook frontends): use the default.
        return cfg.user_prompt
    return typed or cfg.user_prompt


def one_line_preview(text, max_chars=260):
    """Collapse text to one line and truncate for terminal readability."""
    single_line = " ".join(text.split())
    if len(single_line) <= max_chars:
        return single_line, False

    keep_head = int(max_chars * 0.7)
    keep_tail = max_chars - keep_head - 3
    return f"{single_line[:keep_head]}...{single_line[-keep_tail:]}", True


#%%
# 1. SETUP MODEL, TOKENIZER & SAFETY CAPS
cfg = DemoConfig()

MODEL_NAME = cfg.model_name
FILLER_REPEAT = cfg.context_filler_repeat
NUM_GENERATED_TOKENS = cfg.num_generated_tokens
STEP_LOG_EVERY = cfg.step_log_every

device = select_device(cfg.device_mode)
user_prompt = resolve_user_prompt(cfg)

print_section("1) Setup: model, prompt, and run configuration")
print(
    c_yellow(
        "This demo compares autoregressive decoding in two modes: \n"
        "(1) without KV cache, where the model recomputes attention over the growing "
        "context each step, \nand (2) with KV cache, where prior keys/values are reused "
        "so each step processes mainly the newest token. \nIt exemplifies the core "
        "inference trade-off: much faster per-token latency in exchange for steadily "
        "growing KV-cache memory usage."
    )
)
print(f"Model: {c_val(MODEL_NAME)}")
print(f"Device: {c_val(str(device))}")
print(f"Generated tokens requested: {c_val(str(NUM_GENERATED_TOKENS))}")

# Show the prompt up front: every number reported below is measured on this input.
prompt_view, prompt_view_truncated = one_line_preview(user_prompt)
print(f"\n{c_bold('Prompt the model will continue')} {c_dim('(free text -- DemoConfig section A)')}")
print(f"  {c_val(prompt_view)}")
if prompt_view_truncated:
    print(c_dim("  (shown truncated for readability)"))
if FILLER_REPEAT > 0:
    print(
        c_dim(
            f"  Prepended filler context: up to {FILLER_REPEAT} x "
            f'"{cfg.context_filler_sentence.strip()}"'
        )
    )
    print(c_dim("  Filler only lengthens the context; your prompt stays last, so the model continues it."))
else:
    print(c_dim("  No filler context (context_filler_repeat = 0): short-prompt run."))

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# Construct the model input: filler context first, the user's free-text prompt
# LAST, so the generated continuation is conditioned directly on the user text.
user_ids = tokenizer.encode(user_prompt, return_tensors="pt")
filler_ids = (
    tokenizer.encode(cfg.context_filler_sentence * FILLER_REPEAT, return_tensors="pt")
    if FILLER_REPEAT > 0
    else torch.empty((1, 0), dtype=user_ids.dtype)
)

# Trim the FILLER (never the user prompt) so the prompt survives the context cap.
filler_budget = max(0, prompt_token_budget(model, NUM_GENERATED_TOKENS) - user_ids.shape[1])
filler_ids = filler_ids[:, :filler_budget]

input_ids = torch.cat([filler_ids, user_ids], dim=-1).to(device)
if input_ids.shape[1] == 0:
    raise ValueError(
        "Empty prompt: set cfg.user_prompt to some text, or cfg.context_filler_repeat > 0."
    )

# Enforce dynamic safety cap to prevent context overflow crashes
# (only bites if the user prompt alone overflows the model's context window).
input_ids, max_model_ctx = apply_safety_cap(model, input_ids, NUM_GENERATED_TOKENS)

prompt_len = input_ids.shape[1]
batch_size = input_ids.shape[0]
user_prompt_len = min(user_ids.shape[1], prompt_len)
filler_len = prompt_len - user_prompt_len

# Model structural parameters for theoretical memory calculation
n_layers = int(getattr(model.config, "n_layer", getattr(model.config, "num_hidden_layers", -1)))
n_heads = int(getattr(model.config, "n_head", getattr(model.config, "num_attention_heads", -1)))
n_kv_heads = int(getattr(model.config, "num_key_value_heads", n_heads))
hidden_size = int(getattr(model.config, "n_embd", getattr(model.config, "hidden_size", -1)))
head_dim = hidden_size // n_heads
dtype_bytes = next(model.parameters()).element_size()

print(f"Final safe prompt length: {c_green(str(prompt_len))} tokens (Max model window: {max_model_ctx})")
print(c_dim(f"  = {filler_len} filler tokens + {user_prompt_len} tokens of your prompt"))
print(f"Batch size: {batch_size}")

print_narration("Before benchmarking, let's see how this model's attention heads shape the cache we're about to measure.")
arch_info = compute_attention_architecture(n_layers, n_heads, n_kv_heads, head_dim, dtype_bytes, batch_size)
print_attention_architecture_explanation(n_heads, n_kv_heads, head_dim, arch_info)

print_narration("One housekeeping step before timing anything: warm up CUDA so first-call kernel compilation doesn't skew the benchmark below.")
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
print_section("3) Benchmark A: decoding WITHOUT KV cache")
print_narration("First, the expensive baseline: recompute attention over the entire growing context at every single step.")

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

print_section("4) Benchmark B: decoding WITH KV cache")
print_narration("Now the optimized path: reuse cached Key/Value tensors so each step only processes the newest token.")

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
print_section("5) Memory pass: measuring cache footprint")
print_narration("Repeating the cached run once more, this time sampling the cache's VRAM footprint after every step.")
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
if n_layers > 0 and n_kv_heads > 0 and head_dim > 0:
    theoretical_bytes_per_token = 2 * batch_size * n_layers * (n_kv_heads * head_dim) * dtype_bytes
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
print_narration("With both passes complete, here's what the timing and memory numbers say.")
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

print_narration("Zooming into the cached run: its very first step behaves very differently from every step after it.")
prefill_time_ms = times_with_cache[0]
decode_times_ms = times_with_cache[1:]
prefill_throughput, decode_throughput = print_prefill_vs_decode_explanation(
    prompt_len, prefill_time_ms, decode_times_ms
)

print_narration("Distilling everything above into the one trade-off worth remembering.")
print_lecture_takeaway(overall_speedup, late_speedup, per_token_cache_growth_mb)

print_section("9) Model output for your prompt")
print_narration("Here is what the cached run actually generated from the prompt shown in Step 1.")
generated_text = tokenizer.decode(generated_tokens_with_cache, skip_special_tokens=True)

prompt_echo, prompt_echo_truncated = one_line_preview(user_prompt, max_chars=400)
print(f"\n{c_yellow('Your prompt:')}")
print(textwrap.fill(prompt_echo, width=88, initial_indent="  ", subsequent_indent="  "))
if prompt_echo_truncated:
    print(c_dim("  (shown truncated for readability)"))
if filler_len > 0:
    print(c_dim(f"  (preceded by {filler_len} filler tokens of synthetic context)"))

print(f"\n{c_yellow(f'Continuation ({NUM_GENERATED_TOKENS} tokens):')}")
print(
    c_green(
        textwrap.fill(
            generated_text.strip(), width=88, initial_indent="  ", subsequent_indent="  "
        )
    )
)
print(
    c_dim(
        "\nDecoding is greedy (argmax), so this output is deterministic -- which is "
        "exactly what lets the cached and uncached runs be compared token-for-token above."
    )
)

# ==============================================================================
# BONUS: QUANTIZED KV CACHE (optional -- may not be used)
# ==============================================================================
print_section("10) BONUS: Quantized KV Cache Memory Comparison (optional)")
if not cfg.run_quantized_cache_bonus:
    print(c_dim("Skipped by default -- set cfg.run_quantized_cache_bonus = True to run it (see Step 11)."))
else:
    print_narration("As a bonus, let's see how much smaller the cache gets if we quantize it instead of just growing it.")
    print(
        c_yellow(
            f"Attempting a {cfg.quantized_cache_nbits}-bit quantized KV cache via the "
            f"'{cfg.quantized_cache_backend}' backend. This is an optional bonus -- it "
            "requires an extra dependency and is skipped cleanly if it isn't installed."
        )
    )
    try:
        from transformers.cache_utils import QuantizedCache

        # Capped for speed: quantize/dequantize overhead makes this slower than the main pass.
        bonus_steps = min(40, NUM_GENERATED_TOKENS)
        quantized_cache = QuantizedCache(
            backend=cfg.quantized_cache_backend,
            config=model.config,
            nbits=cfg.quantized_cache_nbits,
        )
        curr_input = input_ids.clone()
        quantized_memory_mb = []
        with torch.no_grad():
            for _ in range(bonus_steps):
                outputs = model(curr_input, past_key_values=quantized_cache, use_cache=True)
                quantized_cache = outputs.past_key_values
                quantized_memory_mb.append(get_quantized_cache_memory_mb(quantized_cache))
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                curr_input = next_token

        print_quantized_cache_comparison(
            cfg.quantized_cache_backend,
            cfg.quantized_cache_nbits,
            memory_growth_mb[bonus_steps - 1],
            quantized_memory_mb[-1],
            bonus_steps,
        )
    except ImportError as e:
        print(c_yellow(f"[BONUS SKIPPED] Missing optional dependency: {e}"))
    except Exception as e:
        print(c_yellow(f"[BONUS SKIPPED] Quantized cache demo failed: {e}"))

print_try_this_next(cfg, arch_info)

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

# Add prefill-vs-decode box on Plot 1
ax1.text(
    0.98,
    0.02,
    (
        f"Prefill: {prefill_time_ms:.1f} ms ({prefill_throughput:,.0f} tok/s, compute-bound)\n"
        f"Decode avg: {mean(decode_times_ms):.2f} ms/token "
        f"({decode_throughput:,.1f} tok/s, memory-bound)"
    ),
    transform=ax1.transAxes,
    horizontalalignment="right",
    verticalalignment="bottom",
    fontsize=8.5,
    bbox=dict(boxstyle="round", facecolor="#eef6ff", edgecolor="#0275d8", alpha=0.9),
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

# Add MHA/GQA/MQA architecture box on Plot 2
ax2.text(
    0.98,
    0.02,
    (
        f"{arch_info['attn_type']}: {n_heads} query heads -> {n_kv_heads} KV heads "
        f"({arch_info['group_size']}x sharing)\n"
        f"Cache is {arch_info['savings_ratio']:.1f}x smaller than an MHA model of this size"
    ),
    transform=ax2.transAxes,
    horizontalalignment="right",
    verticalalignment="bottom",
    fontsize=8.5,
    bbox=dict(boxstyle="round", facecolor="#fff8e6", edgecolor="#f0ad4e", alpha=0.9),
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