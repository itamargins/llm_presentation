# KV Cache Memory Analysis & Mathematical Derivations

### Hardware Reference Scales

To put the calculations below into hardware context, keep these standard GPU memory (VRAM) limits in mind:
* **Consumer GPUs (e.g., NVIDIA RTX 4090):** **24 GB VRAM**
* **Enterprise GPUs (e.g., NVIDIA A100 / H100):** **80 GB VRAM**
* **Next-Gen Accelerators (e.g., NVIDIA B200):** **192 GB VRAM**

When evaluating batch serving ($b=16$), remember that the KV cache must fit in VRAM alongside the static model weights (~140 GB for Llama-3 70B in BF16).

---

## 1. General Formulas & Parameters

The memory required to store the KV cache in high-bandwidth memory (HBM) is calculated as:

$$\text{KV Cache Bytes} = 2 \times b \times s \times l \times h_{kv} \times d_h \times P$$

### Variable Definitions

* **$2$**: Accounts for storing both **Key ($K$)** and **Value ($V$)** matrices.
* **$b$**: Batch size (number of concurrent sequences).
* **$s$**: Sequence length (context length + generated tokens).
* **$l$**: Number of hidden layers.
* **$h_{kv}$**: Number of Key/Value heads per layer.
* **$d_h$**: Dimension of each head ($d_{\text{head}}$).
* **$P$**: Precision in bytes (e.g., $P = 2$ for FP16/BF16, $P = 1$ for INT8/FP8).

---

### Reference Architecture (Llama-3 70B Benchmark)

All comparisons below use the structural configuration of **Llama-3 70B**:

| Parameter | Symbol | Value |
| :--- | :--- | :--- |
| **Layers** | $l$ | 80 |
| **Query Heads** | $h_q$ | 64 |
| **Head Dimension** | $d_h$ | 128 |
| **Precision** | $P$ | 2 bytes (BF16) |
| **Sequence Length** | $s$ | 8,192 tokens |
| **Batch Size** | $b$ | 16 |

---

## 2. Multi-Head Attention (MHA)

In standard MHA, every Query head has a corresponding Key head and Value head ($h_{kv} = h_q = 64$).

### Step-by-Step Derivation

1. **Memory per Token (Single Layer):**
   $$\text{Memory}_{\text{1-token, 1-layer}} = 2 \times 1 \times 64 \times 128 \times 2 = 32,768 \text{ bytes} = 32 \text{ KB}$$

2. **Memory per Token (All 80 Layers):**
   $$\text{Memory}_{\text{1-token, 80-layers}} = 32,768 \text{ bytes} \times 80 = 2,621,440 \text{ bytes} \approx 2.62 \text{ MB}$$

3. **Memory per Sequence ($s = 8,192$, $b = 1$):**
   $$\text{Memory}_{\text{1-seq}} = 2,621,440 \text{ bytes} \times 8,192 = 21,474,836,480 \text{ bytes} \approx 21.47 \text{ GB}$$

4. **Total Cache Footprint ($b = 16$, $s = 8,192$):**
   $$\text{Total MHA Cache} = 2 \times 16 \times 8,192 \times 80 \times 64 \times 128 \times 2 = 343,597,383,680 \text{ bytes} = \mathbf{343.60 \text{ GB}}$$

> **Takeaway (MHA):** **Astronomically large.** A single sequence alone takes ~21.5 GB, which completely fills a consumer RTX 4090 (24 GB) before even loading model weights. At a batch size of 16, **343.60 GB** exceeds the capacity of four enterprise H100 GPUs (80 GB each) combined purely for the cache, making standard MHA unusable for long-context batch serving.  
> **Context Length Limits:** Practical context lengths are severely capped at **~4k to 8k tokens** max (even on multi-GPU setups), as pushing to 32k tokens would require ~86 GB per user just for the cache.

---

## 3. Grouped-Query Attention (GQA)

GQA groups Query heads into clusters where multiple Query heads share a single $K/V$ head pair. For Llama-3 70B, $h_q = 64$ query heads are mapped to $h_{kv} = 8$ key/value heads (Group ratio $g = 8$).

### Step-by-Step Derivation

1. **Memory per Token (Single Layer):**
   $$\text{Memory}_{\text{1-token, 1-layer}} = 2 \times 1 \times 8 \times 128 \times 2 = 4,096 \text{ bytes} = 4 \text{ KB}$$

2. **Memory per Token (All 80 Layers):**
   $$\text{Memory}_{\text{1-token, 80-layers}} = 4,096 \text{ bytes} \times 80 = 327,680 \text{ bytes} \approx 0.33 \text{ MB}$$

3. **Total Cache Footprint ($b = 16$, $s = 8,192$):**
   $$\text{Total GQA Cache} = 2 \times 16 \times 8,192 \times 80 \times 8 \times 128 \times 2 = 42,949,672,960 \text{ bytes} = \mathbf{42.95 \text{ GB}}$$

4. **Memory Reduction:**
   $$\text{Reduction Factor} = \frac{343.60 \text{ GB}}{42.95 \text{ GB}} = \mathbf{8\times \text{ reduction vs. MHA}}$$

> **Takeaway (GQA):** **Practical and deployment-ready.** An $8\times$ memory reduction lowers the batch footprint from ~344 GB to **42.95 GB**. This allows a tensor-parallel node of enterprise GPUs to comfortably fit both the 70B model weights and a multi-user context cache. This balance of quality and memory efficiency is why GQA is the modern standard (e.g., Llama-3).  
> **Context Length Limits:** Supports practical context windows of **32k to 128k tokens**. For instance, a single user on a dedicated 80 GB GPU node can reach ~128k tokens (~42 GB cache), while multi-user batch production realistically operates in the **16k to 32k token** range.

---

## 4. Multi-Query Attention (MQA)

MQA uses a single Key head and a single Value head across all Query heads in a layer ($h_{kv} = 1$).

### Step-by-Step Derivation

1. **Memory per Token (Single Layer):**
   $$\text{Memory}_{\text{1-token, 1-layer}} = 2 \times 1 \times 1 \times 128 \times 2 = 512 \text{ bytes}$$

2. **Memory per Token (All 80 Layers):**
   $$\text{Memory}_{\text{1-token, 80-layers}} = 512 \text{ bytes} \times 80 = 40,960 \text{ bytes} \approx 40.96 \text{ KB}$$

3. **Total Cache Footprint ($b = 16$, $s = 8,192$):**
   $$\text{Total MQA Cache} = 2 \times 16 \times 8,192 \times 80 \times 1 \times 128 \times 2 = 5,368,709,120 \text{ bytes} = \mathbf{5.37 \text{ GB}}$$

4. **Memory Reduction:**
   $$\text{Reduction Factor} = \frac{343.60 \text{ GB}}{5.37 \text{ GB}} = \mathbf{64\times \text{ reduction vs. MHA}}$$

> **Takeaway (MQA):** **Extreme compression.** Reducing the cache to **5.37 GB** ($64\times$ smaller than MHA) minimizes memory bandwidth bottlenecks and frees up massive VRAM headroom for large throughput. However, forcing all 64 query heads to share a single KV head can degrade model quality and reasoning capacity on complex tasks.  
> **Context Length Limits:** Opens up massive context windows of **128k to 512k+ tokens**. A single user single-sequence cache at 128k tokens takes only ~5.2 GB, enabling long-context processing even under tight VRAM budgets.

---

## 5. Multi-Head Latent Attention (MLA)

Introduced in architectures like DeepSeek-V2/V3, MLA compresses Keys and Values into a shared low-rank latent vector $\mathbf{c}_t^{KV}$ ($d_c = 512$) along with a decoupled RoPE positional key vector $\mathbf{k}_t^R$ ($d_R = 64$). 

Instead of caching $K$ and $V$ matrices per head, MLA caches only the latent compression vectors:

$$\text{KV Cache Bytes (MLA)} = b \times s \times l \times (d_c + d_R) \times P$$

*(Note: The factor of 2 is removed because Keys and Values are jointly compressed into one representation).*

### Step-by-Step Derivation ($d_c = 512$, $d_R = 64$)

1. **Memory per Token (Single Layer):**
   $$\text{Memory}_{\text{1-token, 1-layer}} = 1 \times (512 + 64) \times 2 = 1,152 \text{ bytes}$$

2. **Memory per Token (All 80 Layers):**
   $$\text{Memory}_{\text{1-token, 80-layers}} = 1,152 \text{ bytes} \times 80 = 92,160 \text{ bytes} \approx 92.16 \text{ KB}$$

3. **Total Cache Footprint ($b = 16$, $s = 8,192$):**
   $$\text{Total MLA Cache} = 16 \times 8,192 \times 80 \times 576 \times 2 = 12,079,595,520 \text{ bytes} = \mathbf{12.08 \text{ GB}}$$

4. **Memory Reduction:**
   $$\text{Reduction Factor} = \frac{343.60 \text{ GB}}{12.08 \text{ GB}} \approx \mathbf{28.4\times \text{ reduction vs. MHA}}$$

> **Takeaway (MLA):** **Best of both worlds.** At **12.08 GB** (~$28.4\times$ reduction vs. MHA), MLA approaches MQA-level cache efficiency while retaining the full expressiveness of MHA during matrix multiplication by projecting latents back on the fly. This architecture unlocks ultra-long context lengths and high concurrent batch sizes on standard server clusters.  
> **Context Length Limits:** Easily scales to **128k to 1M+ tokens**. At a 128k sequence length, a single user's MLA cache consumes only ~11.8 GB, allowing multi-user production clusters to serve 100k+ token prompts at scale without losing representation quality.

---

## Summary Comparison Table

Below is the consolidated footprint analysis for $b = 16$ and $s = 8,192$ using BF16 ($P = 2$):

| Variant | KV Heads ($h_{kv}$) | Cache Bytes / Token / Layer | Total Cache Size ($b=16, s=8,192$) | Relative Size | Practical Context Ceiling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MHA** | 64 | 32,768 B (32 KB) | **343.60 GB** | 100% | **4k – 8k tokens** |
| **GQA** *(Llama-3)* | 8 | 4,096 B (4 KB) | **42.95 GB** | 12.5% | **32k – 128k tokens** |
| **MLA** *(DeepSeek)* | Latent ($d_c=512$) | 1,152 B (1.15 KB) | **12.08 GB** | ~3.5% | **128k – 1M+ tokens** |
| **MQA** | 1 | 512 B (0.5 KB) | **5.37 GB** | ~1.56% | **128k – 512k+ tokens** |

---

## Quick Reference Code

```python
def calculate_kv_cache_gb(
    batch_size: int,
    seq_len: int,
    layers: int,
    num_kv_heads: int,
    head_dim: int,
    precision_bytes: int = 2,
) -> float:
    """Calculates KV cache size in Gigabytes (GB) for standard MHA/GQA/MQA."""
    total_bytes = (
        2 * batch_size * seq_len * layers * num_kv_heads * head_dim * precision_bytes
    )
    return total_bytes / (1024**3)


# Example: Llama-3 70B with GQA
gqa_size = calculate_kv_cache_gb(
    batch_size=16, seq_len=8192, layers=80, num_kv_heads=8, head_dim=128
)
print(f"Llama-3 70B (GQA) Cache Size: {gqa_size:.2f} GB")