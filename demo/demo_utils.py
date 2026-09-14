"""
demo_utils.py
Helper functions, ANSI colors, memory calculators, and descriptive logging for KV Cache Demo.
"""

import torch


# ==============================================================================
# ANSI COLOR UTILITIES
# ==============================================================================
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"


def c_title(text):
    return f"{Colors.OKCYAN}{Colors.BOLD}{text}{Colors.ENDC}"

def c_header(text):
    return f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}"

def c_green(text):
    return f"{Colors.OKGREEN}{Colors.BOLD}{text}{Colors.ENDC}"

def c_red(text):
    return f"{Colors.FAIL}{Colors.BOLD}{text}{Colors.ENDC}"

def c_yellow(text):
    return f"{Colors.WARNING}{text}{Colors.ENDC}"

def c_bold(text):
    return f"{Colors.BOLD}{text}{Colors.ENDC}"

def c_val(text):
    return f"{Colors.OKBLUE}{Colors.BOLD}{text}{Colors.ENDC}"

def c_dim(text):
    return f"{Colors.DIM}{text}{Colors.ENDC}"


# ==============================================================================
# PRINTING & FORMATTING HELPERS
#
# Color convention used throughout this demo's output:
#   cyan    (print_section)  -- structural markers: "we are now in phase N"
#   magenta (print_narration) -- connective narration: what's happening now and why
#   green / red               -- findings: measured results, good vs. bad
#   blue    (c_val)           -- neutral measured/config values
#   yellow  (c_yellow)        -- warnings, caveats, optional/skippable steps
#   dim     (c_dim)           -- secondary notes and descriptions
# ==============================================================================
def print_section(title):
    print("\n" + c_title("=" * 95))
    print(c_title(f" {title} "))
    print(c_title("=" * 95))


def print_narration(text):
    """One-line connective narration explaining what is about to happen and why,
    printed between a section header and its content to keep the run readable in order."""
    print(f"{Colors.HEADER}▸ {text}{Colors.ENDC}")


def should_log_step(step_idx, total_steps, every):
    if step_idx <= 5 or step_idx == total_steps:
        return True
    return step_idx % every == 0


def print_step(prefix, step_idx, total_steps, input_len, step_ms, extra=""):
    color = c_red if "No cache" in prefix else c_green
    print(
        f"{color(prefix)} step {step_idx:>3}/{total_steps} | "
        f"model_input_len={input_len:>4} | "
        f"step_time={step_ms:>8.2f} ms{extra}"
    )


def print_checkpoint_table(prompt_len, times_no_cache, times_with_cache, memory_growth_mb):
    checkpoints = [1, 2, 3, 5, 10, 20, 40, 60, 80, 100]
    checkpoints = [s for s in checkpoints if s <= len(times_no_cache)]

    print("\n" + c_header("Checkpoint Table (Selected Decode Steps Across Generation)"))
    print(
        f"{c_bold('step'):>13} | {'ctx_len':>7} | {'no_cache(ms)':>12} | "
        f"{'with_cache(ms)':>14} | {'speedup':>16} | {'cache_mb':>8}"
    )
    print("-" * 80)

    for step in checkpoints:
        i = step - 1
        ctx_len = prompt_len + i
        no_cache_ms = times_no_cache[i]
        with_cache_ms = times_with_cache[i]
        speedup = no_cache_ms / with_cache_ms if with_cache_ms > 0 else float("inf")
        cache_mb = memory_growth_mb[i]
        
        speedup_str = c_green(f"{speedup:>7.2f}x") if speedup > 1.2 else c_yellow(f"{speedup:>7.2f}x")
        
        print(
            f"{step:>4} | {ctx_len:>7} | {no_cache_ms:>12.2f} | "
            f"{with_cache_ms:>14.2f} | {speedup_str} | {cache_mb:>8.2f}"
        )


def compute_attention_architecture(n_layers, n_heads, n_kv_heads, head_dim, dtype_bytes, batch_size):
    """Classify the attention head layout (MHA/GQA/MQA) and quantify its cache-size effect.

    The KV cache stores one Key/Value pair per KV head, not per query head, so
    grouping query heads onto fewer KV heads (GQA/MQA) shrinks the cache
    proportionally, independent of the time/memory benchmarks elsewhere in this file.
    """
    if n_kv_heads >= n_heads:
        attn_type = "MHA (Multi-Head Attention)"
        group_size = 1
    elif n_kv_heads == 1:
        attn_type = "MQA (Multi-Query Attention)"
        group_size = n_heads
    else:
        attn_type = "GQA (Grouped-Query Attention)"
        group_size = n_heads // n_kv_heads

    actual_bytes_per_token = 2 * batch_size * n_layers * (n_kv_heads * head_dim) * dtype_bytes
    hypothetical_mha_bytes_per_token = 2 * batch_size * n_layers * (n_heads * head_dim) * dtype_bytes
    savings_ratio = (
        hypothetical_mha_bytes_per_token / actual_bytes_per_token if actual_bytes_per_token > 0 else 1.0
    )

    return {
        "attn_type": attn_type,
        "group_size": group_size,
        "actual_bytes_per_token": actual_bytes_per_token,
        "hypothetical_mha_bytes_per_token": hypothetical_mha_bytes_per_token,
        "savings_ratio": savings_ratio,
    }


def print_attention_architecture_explanation(n_heads, n_kv_heads, head_dim, arch_info):
    attn_type = arch_info["attn_type"]
    group_size = arch_info["group_size"]
    actual_kb = arch_info["actual_bytes_per_token"] / 1024
    hypothetical_kb = arch_info["hypothetical_mha_bytes_per_token"] / 1024
    savings_ratio = arch_info["savings_ratio"]

    print_section("2) Why Cache Size Depends on Architecture: MHA vs GQA vs MQA")

    head_word = "query head" if group_size == 1 else "query heads"
    print(c_bold("The cause:"))
    print(
        "  The KV cache stores one Key/Value pair per KV head, not per "
        f"query head. In this model, {c_bold(str(group_size))} {head_word} "
        "share a single K/V head."
    )
    print()
    print(f"{c_bold('Detected architecture:')} {c_val(attn_type)}")
    print(f"  • Query heads (n_heads)         : {c_val(str(n_heads))}")
    print(f"  • KV heads (n_kv_heads)         : {c_val(str(n_kv_heads))}")
    print(f"  • Head dimension                : {c_val(str(head_dim))}")
    print(f"  • Query heads sharing 1 KV head : {c_val(f'{group_size}x')}")
    print()

    print(c_bold("Effect on cache size (per cached token, all layers):"))
    print(f"  • Actual cache size/token                  : {c_val(f'{actual_kb:.2f} KB')}")
    print(f"  • If this were full MHA (same hidden size) : {c_val(f'{hypothetical_kb:.2f} KB')}")
    if savings_ratio > 1.01:
        print(
            f"  {c_green(f'=> {attn_type} shrinks the KV cache by {savings_ratio:.1f}x')} "
            "vs. an MHA model of the same size -- this is why production LLMs "
            "(Llama, Qwen, Mistral, ...) use GQA/MQA instead of full MHA."
        )
    else:
        print(f"  {c_dim('=> This model uses full MHA: no KV-head sharing reduction applies.')}")


def print_prefill_vs_decode_explanation(prompt_len, prefill_ms, decode_times_ms):
    """Split the 'with cache' run into its prefill step and decode steps and explain why
    they sit in different performance regimes (compute-bound vs memory-bandwidth-bound)."""
    avg_decode_ms = sum(decode_times_ms) / len(decode_times_ms) if decode_times_ms else float("nan")
    prefill_throughput = prompt_len / (prefill_ms / 1000) if prefill_ms > 0 else float("inf")
    decode_throughput = 1000 / avg_decode_ms if avg_decode_ms > 0 else float("inf")

    print_section("7) Prefill vs. Decode: Two Different Computational Regimes")
    print(c_bold("Prefill (this run's first step):"))
    print(
        f"  Processes the entire {c_val(str(prompt_len))}-token prompt in a single "
        "forward pass. All positions are computed in parallel, so the model's "
        "weights are reused across many tokens per read from VRAM."
    )
    print(f"  • Prefill time       : {c_val(f'{prefill_ms:.2f} ms')}")
    print(f"  • Prefill throughput : {c_val(f'{prefill_throughput:,.0f} tokens/sec')} {c_dim('(compute-bound)')}")
    print()
    print(c_bold("Decode (every step after the first):"))
    print(
        "  Processes exactly one new token per step, reusing cached K/V. Each "
        "step still reads the full model weights (and the growing cache) from "
        "VRAM to produce a single token, so GPU compute sits mostly idle "
        "waiting on memory bandwidth."
    )
    print(f"  • Avg decode step time : {c_val(f'{avg_decode_ms:.2f} ms')}")
    print(f"  • Decode throughput    : {c_val(f'{decode_throughput:,.1f} tokens/sec')} {c_dim('(memory-bandwidth-bound)')}")
    print()
    print(
        f"  {c_yellow('=> Not directly comparable speeds:')} prefill parallelizes "
        "over many tokens per weight-read, while decode pays a full weight-read "
        "for just one token -- this is why decode throughput, not prefill, "
        "dominates real-world serving cost."
    )
    return prefill_throughput, decode_throughput


def print_quantized_cache_comparison(backend, nbits, baseline_mb, quantized_mb, num_steps):
    print(c_bold(f"Result after {num_steps} decode steps ({backend} backend, {nbits}-bit):"))
    print(f"  • Baseline (unquantized) cache size : {c_val(f'{baseline_mb:.2f} MB')}")
    print(f"  • Quantized cache size              : {c_val(f'{quantized_mb:.2f} MB')}")
    if quantized_mb > 0:
        ratio = baseline_mb / quantized_mb
        print(f"  {c_green(f'=> ~{ratio:.1f}x smaller cache at {nbits}-bit')} vs. the unquantized baseline.")
    print(
        c_dim(
            "  Note: this is a best-effort measurement of whatever tensor storage the "
            "installed backend exposes, so treat the ratio as approximate rather than exact."
        )
    )


def print_descriptive_results(
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
):
    print_section("6) Quantitative Summary: Benchmark Metrics & Descriptions")

    def log_entry(label, val_str, description):
        print(f"• {c_bold(label):<42} : {val_str:<22} {c_dim('--> ' + description)}")

    # Time Metrics
    log_entry(
        "Total time without cache",
        c_red(f"{total_no_cache_ms:,.2f} ms"),
        "Cumulative wall-clock time to generate all tokens re-processing full context each step."
    )
    log_entry(
        "Total time with cache",
        c_green(f"{total_with_cache_ms:,.2f} ms"),
        "Cumulative wall-clock time to generate all tokens using cached Key/Value tensors."
    )
    log_entry(
        "Overall speedup from KV cache",
        c_green(f"{overall_speedup:,.2f}x faster"),
        "Total speed multiplier gained across the entire generation run."
    )
    print()

    log_entry(
        "Average step time without cache",
        c_red(f"{avg_no_cache_ms:,.2f} ms"),
        "Mean latency per individual token generated without KV caching."
    )
    log_entry(
        "Average step time with cache",
        c_green(f"{avg_with_cache_ms:,.2f} ms"),
        "Mean latency per individual token generated with KV caching."
    )
    log_entry(
        f"Late-token speedup (last {late_window} steps)",
        c_green(f"{late_speedup:,.2f}x faster"),
        "Speedup at the end of prompt, where quadratic context re-computation penalty is highest."
    )
    print()

    # Growth Ratio Descriptions
    log_entry(
        "No-cache step-time growth (last/first)",
        c_red(f"{no_cache_growth:,.2f}x slower"),
        "Ratio of final token time vs 1st token time without cache (shows computational slowdown as context grows)."
    )
    log_entry(
        "With-cache step-time growth (last/first)",
        c_green(f"{with_cache_growth:,.2f}x change"),
        "Ratio of final token time vs 1st token time with cache (ideally ~1.0x, confirming O(1) step latency)."
    )
    print()

    # Memory Metrics
    log_entry(
        "KV cache footprint after prefill",
        c_val(f"{prefill_cache_mb:,.2f} MB"),
        "VRAM allocated to cache initial prompt tokens before generating new text."
    )
    log_entry(
        "KV cache footprint at end",
        c_val(f"{peak_cache_mb:,.2f} MB"),
        "Total VRAM consumed by prompt + all newly generated token KV pairs."
    )
    log_entry(
        "Measured cache growth per token",
        c_val(f"{per_token_cache_growth_mb:,.4f} MB/token"),
        "Empirical VRAM added to the cache for each newly generated token."
    )

    if theoretical_mb_per_token is not None:
        log_entry(
            "Theoretical growth per token",
            c_val(f"{theoretical_mb_per_token:,.4f} MB/token"),
            "Calculated exact tensor size: 2 * batch * layers * (kv_heads * head_dim) * dtype_bytes "
            "-- uses KV heads, not query heads, so it stays exact for GQA/MQA models too."
        )

    print()
    match_status = c_green("TRUE (Exact Match)") if token_match else c_red("FALSE (Mismatch)")
    log_entry(
        "Token-by-token output match",
        match_status,
        "Verifies that caching produces mathematically identical outputs vs raw recompute."
    )

    print_checkpoint_table(prompt_len, times_no_cache, times_with_cache, memory_growth_mb)


def print_lecture_takeaway(overall_speedup, late_speedup, per_token_cache_growth_mb):
    print_section("8) Lecture Takeaway")
    print(c_header("THE CORE TRADE-OFF:"))
    print(f"1. {c_bold('Speed:')} KV Cache converts an {c_red('O(N^2)')} context recompute problem into an {c_green('O(1)')} step time decode process.")
    print(f"2. {c_bold('Memory:')} The price paid is VRAM growth of ~{c_val(f'{per_token_cache_growth_mb:.4f} MB')} per token generated.")
    print(f"3. {c_bold('Speedup Achieved:')} {c_green(f'{overall_speedup:.2f}x total speedup')} ({c_green(f'{late_speedup:.2f}x speedup')} on late tokens).")


def print_try_this_next(cfg, arch_info):
    """Closing suggestions for what to change in DemoConfig before re-running the demo."""
    print_section("11) Try This Next: Parameters Worth Tweaking")
    print(
        c_dim(
            "All of these live in DemoConfig at the top of kv_cache.py -- change one, "
            "re-run, and compare against this run."
        )
    )
    print()

    def suggestion(param, current, idea):
        print(f"• {c_bold(param)} {c_dim('(currently')} {c_val(str(current))}{c_dim(')')}")
        print(f"    {c_dim(idea)}")
        print()

    suggestion(
        "model_name",
        cfg.model_name,
        "Switch to 'Qwen/Qwen2.5-0.5B' or 'meta-llama/Llama-3.2-1B' to see Step 2's "
        f"GQA cache savings actually kick in -- this run used {arch_info['attn_type']}, "
        f"so the savings ratio above was {arch_info['savings_ratio']:.1f}x.",
    )
    suggestion(
        "prompt_repeat",
        cfg.prompt_repeat,
        "Raise it to lengthen the initial context and make the no-cache quadratic "
        "slowdown in Step 3 more dramatic; lower it for a faster run.",
    )
    suggestion(
        "num_generated_tokens",
        cfg.num_generated_tokens,
        "Raise it to watch the cache in Step 5 grow further and see how long the "
        "O(1) decode latency from Step 7 holds up.",
    )
    suggestion(
        "run_quantized_cache_bonus",
        cfg.run_quantized_cache_bonus,
        "Set to True (with `pip install optimum-quanto` or `pip install hqq`) to "
        "run Step 10 and see real KV-cache compression from quantization.",
    )
    suggestion(
        "device_mode",
        cfg.device_mode,
        "Force 'cpu' to compare against this run's device -- the cache's memory tax "
        "stays the same, but the compute-bound vs. memory-bound gap from Step 7 "
        "looks very different without a GPU's memory bandwidth.",
    )


# ==============================================================================
# HARDWARE & MODEL UTILITIES
# ==============================================================================
def sync_device(device):
    """Synchronize CUDA stream if running on GPU to ensure accurate timings."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def select_device(device_mode):
    if device_mode == "cpu":
        return torch.device("cpu")
    if device_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_kv_cache_memory_mb(past_key_values):
    """Calculate the memory footprint in MB of the active past_key_values structure."""
    if past_key_values is None:
        return 0.0

    total_bytes = 0
    # DynamicCache/QuantizedCache check
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        if len(past_key_values.key_cache) > 0 and isinstance(past_key_values.key_cache[0], torch.Tensor):
            for k_tensor, v_tensor in zip(past_key_values.key_cache, past_key_values.value_cache):
                total_bytes += k_tensor.nelement() * k_tensor.element_size()
                total_bytes += v_tensor.nelement() * v_tensor.element_size()
            return total_bytes / (1024 ** 2)

    # Tuple / Layer object fallback
    if hasattr(past_key_values, "__len__"):
        for layer in past_key_values:
            if isinstance(layer, (tuple, list)):
                for item in layer:
                    if isinstance(item, torch.Tensor):
                        total_bytes += item.nelement() * item.element_size()
            elif hasattr(layer, "keys") and hasattr(layer, "values"):
                if isinstance(layer.keys, torch.Tensor):
                    total_bytes += layer.keys.nelement() * layer.keys.element_size()
                if isinstance(layer.values, torch.Tensor):
                    total_bytes += layer.values.nelement() * layer.values.element_size()

    return total_bytes / (1024 ** 2)


def get_quantized_cache_memory_mb(past_key_values):
    """Best-effort memory footprint (MB) of a quantized Cache (e.g. transformers' QuantizedCache).

    Sums the small full-precision residual buffer (`.keys`/`.values`) plus whatever
    compressed storage the backend exposes (`_quantized_keys`/`_quantized_values`).
    Quantized wrapper tensors (e.g. optimum-quanto's WeightQBitsTensor) report
    `.element_size()`/`.nelement()` for their ORIGINAL logical dtype, not their packed
    physical storage, so this drills into the wrapper's internal `._data` tensor (plus
    any `_scale`/`_shift`/`_zeropoint` overhead) when present. The exact packed layout
    is still backend-specific, so this is a best-effort measurement rather than a
    guaranteed-exact figure -- unlike `get_kv_cache_memory_mb`, which is exact for
    plain (unquantized) caches.
    """
    if past_key_values is None:
        return 0.0

    def _tensor_bytes(obj):
        if not isinstance(obj, torch.Tensor) or obj.nelement() == 0:
            return 0

        packed = getattr(obj, "_data", None)
        if isinstance(packed, torch.Tensor) and packed.nelement() > 0:
            total = packed.nelement() * packed.element_size()
            for meta_attr in ("_scale", "_shift", "_zeropoint"):
                meta = getattr(obj, meta_attr, None)
                if isinstance(meta, torch.Tensor):
                    total += meta.nelement() * meta.element_size()
            return total

        return obj.nelement() * obj.element_size()

    total_bytes = 0
    layers = getattr(past_key_values, "layers", past_key_values)
    for layer in layers:
        total_bytes += _tensor_bytes(getattr(layer, "keys", None))
        total_bytes += _tensor_bytes(getattr(layer, "values", None))
        total_bytes += _tensor_bytes(getattr(layer, "_quantized_keys", None))
        total_bytes += _tensor_bytes(getattr(layer, "_quantized_values", None))

    return total_bytes / (1024 ** 2)


def apply_safety_cap(model, input_ids, num_generated_tokens):
    """Dynamically clamp prompt length based on model max positions to avoid crashes."""
    max_model_ctx = getattr(
        model.config,
        "max_position_embeddings",
        getattr(model.config, "n_positions", getattr(model.config, "seq_length", 2048)),
    )
    max_allowed_prompt = max_model_ctx - num_generated_tokens - 5

    if input_ids.shape[1] > max_allowed_prompt:
        print(
            f"\n{c_yellow('[SAFETY CAP] Prompt length')} ({input_ids.shape[1]} tokens) exceeds safe limit "
            f"for model max window ({max_model_ctx} tokens)."
        )
        print(f"{c_yellow('[SAFETY CAP] Truncating initial prompt to')} {c_bold(str(max_allowed_prompt))} tokens to prevent lecture crash.\n")
        input_ids = input_ids[:, :max_allowed_prompt]

    return input_ids, max_model_ctx