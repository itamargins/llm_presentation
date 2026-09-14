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
# ==============================================================================
def print_section(title):
    print("\n" + c_title("=" * 95))
    print(c_title(f" {title} "))
    print(c_title("=" * 95))


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
    print_section("3) Quantitative Summary: Benchmark Metrics & Descriptions")

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
            "Calculated exact tensor size: 2 * batch * layers * hidden_dim * dtype_bytes."
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
    print_section("4) Lecture Takeaway")
    print(c_header("THE CORE TRADE-OFF:"))
    print(f"1. {c_bold('Speed:')} KV Cache converts an {c_red('O(N^2)')} context recompute problem into an {c_green('O(1)')} step time decode process.")
    print(f"2. {c_bold('Memory:')} The price paid is VRAM growth of ~{c_val(f'{per_token_cache_growth_mb:.4f} MB')} per token generated.")
    print(f"3. {c_bold('Speedup Achieved:')} {c_green(f'{overall_speedup:.2f}x total speedup')} ({c_green(f'{late_speedup:.2f}x speedup')} on late tokens).")


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