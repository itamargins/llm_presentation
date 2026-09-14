# SEMINAR PLAN: THE TRANSFORMER BLOCK & KV CACHE MECHANICS

**Target Duration:** ~75–90 Minutes (including Q&A / discussions)

---

## TIMELINE OVERVIEW
+-----------------------------------------------------------------------------------+  
| [00-05m] I.   5-Minute Hand-off: From Attention Output to Residual Vector         |  
| [05-40m] II.  Building the Complete Transformer Block (Functional Rationale)      |  
| [40-60m] III. The Decoding Bottleneck & KV-Cache Algorithmic Mechanics            |  
| [60-75m] IV.  Structural Attention Variants (MQA, GQA, Sliding Window)            |  
| [75-90m] V.   Q&A, Discussion Buffer, & Time Assessment                           |  
+-----------------------------------------------------------------------------------+  

---

## 1. DETAILED LECTURE FLOW

### I. The 5-Minute Hand-off (00:00 – 00:05)

* **Starting State:** Assume $Q, K, V$, scaled dot-product attention, and RoPE were covered in the previous lecture.
* **The Missing Link:** Standard attention outputs a set of context-weighted head vectors. How do we take these independent attention outputs, preserve gradient flow across 80+ stacked layers, process the information within individual tokens, and output next-token probability distributions?

---

### II. Building the Complete Transformer Block: Functional Rationale (00:05 – 00:40)

*Step-by-step assembly of a single causal Transformer layer (using Llama 3 as the primary concrete reference), explaining **why** each component exists:*

#### 1. The Multi-Head Output Projection ($W^O$)
* **Problem:** Multi-head attention generates $h$ separate output vectors ($d_k$ dimension each).
* **Solution:** Concatenate and project back to hidden dimension $d_{\text{model}}$ via $W^O \in \mathbb{R}^{(h \cdot d_k) \times d_{\text{model}}}$.

#### 2. The Residual Stream ("The Shared Bus")
* **Rationale:** 

$$x^{(l)} = x^{(l-1)} + \text{Attention}(x^{(l-1)})$$

* **Functional Insight:** Frame the residual connection not just as a gradient trick, but as a **shared information bus**. The main hidden vector flows through the network unchanged; each block computes a *delta update* ($\Delta x$) and writes it back to the bus.

#### 3. Normalization (Pre-LayerNorm & RMSNorm)
* **Rationale:** Unnormalized residual additions cause activations to blow up in deep networks.
* **Why RMSNorm?** Explain removing mean-centering for computational efficiency without losing scale stability:

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \odot \gamma, \quad \text{where } \text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

#### 4. The Feed-Forward Network (FFN / SwiGLU)
* **Rationale:** Attention *routes* information between tokens; the FFN *processes* information within each token.
* **Functional Insight:** The FFN acts as an **associative memory** (expanding dimension from $d_{\text{model}} \to 4d_{\text{model}} \to d_{\text{model}}$) storing factual concepts and patterns.
* **SwiGLU:** Briefly note why modern models replaced GELU with Gated Linear Units:

$$\text{SwiGLU}(x) = (x W_1) \otimes \text{SiLU}(x W_2) W_3$$

#### 5. The Language Modeling Head (LM Head) & Sampling Loop
* **Rationale:** Projecting the final normalized residual state $x^{(L)}$ back to vocabulary size $V$ using unembedding matrix $W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times V}$, followed by temperature-scaled Softmax to produce the autoregressive generation loop.

---

### III. The Decoding Bottleneck & KV Cache Mechanics (00:40 – 01:00)

* **Prefill vs. Decode Phase:** 
  * **Prefill:** Prompt processing happens in parallel across all prompt tokens ($O(N^2)$ matrix-matrix operations).
  * **Decode:** Generating token $t+1$ requires attending to all previous tokens $1 \dots t$. Running full self-attention at every single token generation step causes redundant $O(N^2)$ re-computations.
* **The KV Cache Algorithmic Solution:**
  * Store previously computed Key ($K$) and Value ($V$) projections in memory across layers.
  * At step $t$, compute $Q_t, K_t, V_t$ for the single new token only. Append $K_t, V_t$ to the cache, and evaluate attention as a matrix-vector operation ($O(N)$ computation per generated token).
* **The Algorithmic / Memory Trade-off:**
  * We trade RAM/VRAM space for compute speed.
  * **Exact Memory Formula (Walk through with concrete numbers):**

$$\text{Memory}_{\text{KVCache}} = 2 \times B \times L \times N_{\text{layers}} \times H_{\text{KV}} \times d_{\text{head}} \times \text{BytesPerPrecision}$$

---

### IV. Structural Attention Variants: Handling KV Cache Limits (01:00 – 01:15)

*How contemporary models alter the attention architecture specifically to prevent the KV cache from exhausting memory during long-context decoding:*

#### 1. Multi-Query Attention (MQA)
* **Concept:** All $H_Q$ Query heads share a **single** $K$ and $V$ head.
* **Trade-off:** Drastic KV cache memory reduction (~$H$-fold decrease), but risks loss of representation capacity.

#### 2. Grouped-Query Attention (GQA - Llama 3 Benchmark)
* **Concept:** The ideal balance. Group $H_Q$ Query heads into $G$ groups, where each group shares 1 Key and 1 Value head.
* **Impact:** Reduces KV cache size by a factor of $H_Q / G$ (e.g., 8x memory savings) while maintaining near-MHA performance.

#### 3. Sliding Window Attention (SWA / Local Attention)
* **Concept:** Token $t$ only attends to tokens within a fixed local window $W$ (e.g., last 4096 tokens).
* **Impact:** Bounds the KV cache memory overhead to $O(W)$ instead of $O(L)$, making context windows effectively infinite from a memory footprint perspective.

---


## 3. CHECKING FOR MISSING LINKS

The flow above is completely self-contained:

* **Link 1 (Attention $\to$ Block):** Resolved by showing how head projection $W^O$ and residual addition $x + f(x)$ turn attention outputs into a continuous residual vector.
* **Link 2 (Routing $\to$ Processing):** Resolved by explaining why attention alone cannot process facts (needs the FFN associative memory).
* **Link 3 (Block $\to$ Autoregressive Loop):** Resolved by showing how the LM Head maps the final block output to token probabilities.
* **Link 4 (Autoregressive Loop $\to$ KV Cache):** Resolved by showing how token-by-token generation creates the memory bandwidth wall during decoding.
* **Link 5 (KV Cache $\to$ MQA/GQA/SWA):** Resolved by showing that standard MHA KV cache memory grows too fast, directly motivating structural architectural variants.