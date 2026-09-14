# III. The Decoding Bottleneck & KV Cache Mechanics

During training, large language models process all tokens in a prompt simultaneously using causal attention masks (the **Prefill Phase**). However, during inference, tokens are generated one at a time in an autoregressive loop (the **Decode Phase**). 

Without intervention, calculating self-attention at step $t$ requires re-projecting and re-computing attention states for all preceding $t-1$ tokens, leading to an $O(N^2)$ computational complexity and severe memory-bandwidth saturation.

---

## 1. Prefill Phase vs. Decode Phase

Understanding the performance transition from prompt ingestion to token generation is essential for diagnosing LLM system bottlenecks.

+-----------------------------------------------------------------------------------+  
| PREFILL PHASE (Prompt Ingestion)                                                  |  
| - Compute Bound (FLOP heavy)                                                      |  
| - Parallel matrix-matrix operations: [B, N, d_model] × [d_model, d_model]         |  
| - High Arithmetic Intensity (FLOPs / Byte loaded)                                 |  
+-----------------------------------------------------------------------------------+  
│  
▼  
+-----------------------------------------------------------------------------------+  
| DECODE PHASE (Autoregressive Generation)                                          |  
| - Memory Bandwidth Bound (VRAM transfer heavy)                                    |  
| - Sequential matrix-vector operations: [B, 1, d_model] × [d_model, d_model]       |  
| - Low Arithmetic Intensity (Model weights re-loaded for every single token)      |  
+-----------------------------------------------------------------------------------+  


### A. The Prefill Phase (Compute-Bound)
Given a prompt of length $N$, the model processes all $N$ tokens concurrently.
* **Operations:** Matrix-Matrix multiplications (GEMM).
* **Execution:** GPUs achieve high Tensor Core utilization because reading weight matrices $W_Q, W_K, W_V, W_O$ from High Bandwidth Memory (HBM) into SRAM is amortized over $N$ tokens simultaneously.
* **Time Complexity:** $O(N^2)$ due to the pairwise attention score matrix $S = Q K^T \in \mathbb{R}^{B \times h \times N \times N}$.

### B. The Decode Phase (Memory-Bandwidth Bound)
To generate token $N+1$, the model takes only the newly sampled token $x_N$ as input.
* **Operations:** Matrix-Vector multiplications (GEMV).
* **Execution:** For every generated token, all billions of weight parameters across all $L$ layers must be transferred from GPU VRAM to compute units (SRAM) to process a single token vector of shape $[B, 1, d_{\text{model}}]$.
* **Arithmetic Intensity:** Extremely low. The GPU spend most of its time waiting for memory transfers rather than executing floating-point arithmetic.

---

## 2. Naive Attention vs. KV Caching

### The Naive Generation Loop (Without Cache)
To generate token $t+1$ given past tokens $x_{1:t}$, naive autoregressive generation re-evaluates the entire sequence through all layers:

$$\text{For step } t: \quad Q_{1:t} = x_{1:t} W_Q, \quad K_{1:t} = x_{1:t} W_K, \quad V_{1:t} = x_{1:t} W_V$$

$$\text{Attention}_t = \text{Softmax}\left(\frac{Q_{1:t} K_{1:t}^T}{\sqrt{d_k}} + M\right) V_{1:t}$$

* **Redundancy:** The Key and Value projections $K_{1:t-1}$ and $V_{1:t-1}$ for past tokens $x_{1:t-1}$ **do not change** as token $t$ is generated. Recomputing them requires $2 \times L \times t$ matrix multiplications per token step.
* **Cumulative Compute Cost:** Generating $N$ tokens requires $\sum_{t=1}^N O(t^2) = O(N^3)$ operations.

Naive Decoding (Step t = 4):
Recompute Token 1: [x_1 -> Q_1, K_1, V_1]  (Redundant!)
Recompute Token 2: [x_2 -> Q_2, K_2, V_2]  (Redundant!)
Recompute Token 3: [x_3 -> Q_3, K_3, V_3]  (Redundant!)
Compute   Token 4: [x_4 -> Q_4, K_4, V_4]


### The KV Cache Solution
Since attention at position $t$ only requires querying past Keys and Values:

$$\text{Attention}_t = \text{Softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$

We only need to compute Query $Q_t$, Key $K_t$, and Value $V_t$ for the **single current token** $x_t$. Past Key and Value projections are retrieved from GPU memory (the **KV Cache**).

KV-Cached Decoding (Step t = 4):
Fetch Cached:     [K_1..3, V_1..3] from VRAM
Compute Current:  [x_4 -> Q_4, K_4, V_4]
Append & Store:   KV_Cache.append(K_4, V_4)
Compute Attention: Softmax(Q_4 · [K_1..3, K_4]^T / sqrt(d_k)) · [V_1..3, V_4]


### Computational Complexity Reduction
* **Per-Token Compute:** Reduced from $O(t)$ matrix-matrix projections to $O(1)$ projections + $O(t)$ vector-matrix dot products.
* **Total Generation Sequence Cost:** Reduced from $O(N^3)$ to $O(N^2)$.

---

## 3. Algorithmic Mechanics of the KV Cache

Let $x_t \in \mathbb{R}^{B \times 1 \times d_{\text{model}}}$ be the normalized hidden vector of the newest token entering layer $l$ at generation step $t$.

### Step-by-Step Execution per Layer

#### Step 1: Projection of Current Token
Compute Query, Key, and Value projections for position $t$ only:

$$Q_t = x_t W_Q \in \mathbb{R}^{B \times 1 \times h_Q \times d_k}$$

$$K_t = x_t W_K \in \mathbb{R}^{B \times 1 \times h_{KV} \times d_k}$$

$$V_t = x_t W_V \in \mathbb{R}^{B \times 1 \times h_{KV} \times d_k}$$

#### Step 2: RoPE Application
Apply Rotary Position Embedding (RoPE) to $Q_t$ and $K_t$ using the current absolute position index $t$:

$$Q_t = \mathbf{R}_{\Theta, t}^d Q_t, \quad K_t = \mathbf{R}_{\Theta, t}^d K_t$$

#### Step 3: Cache Concatenation / Update
Retrieve the cached tensor history $\mathcal{K}_{1:t-1}, \mathcal{V}_{1:t-1} \in \mathbb{R}^{B \times (t-1) \times h_{KV} \times d_k}$ from memory and append current projections:

$$\mathcal{K}_{1:t} = \text{Concat}\left(\mathcal{K}_{1:t-1}, K_t\right) \in \mathbb{R}^{B \times t \times h_{KV} \times d_k}$$

$$\mathcal{V}_{1:t} = \text{Concat}\left(\mathcal{V}_{1:t-1}, V_t\right) \in \mathbb{R}^{B \times t \times h_{KV} \times d_k}$$

#### Step 4: Causal Attention Evaluation
Compute scaled dot-product attention between single Query $Q_t$ and full context Key history $\mathcal{K}_{1:t}$:

$$S_t = \frac{Q_t \mathcal{K}_{1:t}^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times h_Q \times 1 \times t}$$

$$P_t = \text{Softmax}(S_t) \in \mathbb{R}^{B \times h_Q \times 1 \times t}$$

$$O_t = P_t \mathcal{V}_{1:t} \in \mathbb{R}^{B \times h_Q \times 1 \times d_k}$$

#### Step 5: Output Projection
Project multi-head context output back to residual space:

$$\text{Output}_t = O_t W^O \in \mathbb{R}^{B \times 1 \times d_{\text{model}}}$$

---

## 4. KV Cache Memory Footprint: Formula & Concrete Derivation

While KV Caching eliminates re-computation, it converts compute bottlenecks directly into **VRAM consumption bottlenecks**.

### The Exact KV Cache Memory Formula

$$\text{Memory}_{\text{KVCache}} = 2 \times B \times L_{\text{seq}} \times N_{\text{layers}} \times h_{\text{KV}} \times d_{\text{head}} \times \text{BytesPerPrecision}$$

Where:
* $2$: Represents two separate cached matrices ($K$ and $V$).
* $B$: Batch size (number of concurrent requests).
* $L_{\text{seq}}$: Total sequence length ($N_{\text{prompt}} + N_{\text{generated}}$).
* $N_{\text{layers}}$: Total number of Transformer blocks in the network.
* $h_{\text{KV}}$: Number of Key-Value heads per layer.
* $d_{\text{head}}$: Dimension per head ($d_{\text{model}} / h_Q$).
* $\text{BytesPerPrecision}$: Data type size in bytes ($2$ for FP16/BF16, $1$ for INT8, $0.5$ for FP4).

---

### Concrete Numerical Example 1: Standard MHA (Llama 2 70B)

Consider high-throughput serving of a standard Multi-Head Attention model:
* Model Parameters: $70\text{B}$
* Layers ($N_{\text{layers}}$): $80$
* Hidden Dimension ($d_{\text{model}}$): $8192$
* Key-Value Heads ($h_{\text{KV}}$): $64$ (Standard MHA, $h_{\text{KV}} = h_Q$)
* Head Dimension ($d_{\text{head}}$): $8192 / 64 = 128$
* Precision: 16-bit Float ($\text{BF16} = 2 \text{ bytes}$)
* Batch Size ($B$): $32$
* Sequence Length ($L_{\text{seq}}$): $4096 \text{ tokens}$

#### Calculation:
$$\text{Memory}_{\text{KVCache}} = 2 \times 32 \times 4096 \times 80 \times 64 \times 128 \times 2 \text{ bytes}$$

$$\text{Memory}_{\text{KVCache}} = 2 \times 32 \times 4096 \times 80 \times 8192 \times 2$$

$$\text{Memory}_{\text{KVCache}} = 274,877,906,944 \text{ bytes} \approx \mathbf{256 \text{ GB}}$$

> **Insight:** The KV Cache alone requires **256 GB of VRAM**—more than double the memory needed to store the model weights themselves in FP16 (~140 GB). This single bottleneck severely constrains maximum serving batch size.

---

### Concrete Numerical Example 2: Modern GQA Architecture (Llama 3 8B)

Now examine modern architectural optimizations on Llama 3 8B using **Grouped-Query Attention (GQA)**:
* Model Parameters: $8\text{B}$
* Layers ($N_{\text{layers}}$): $32$
* Query Heads ($h_Q$): $32$
* Key-Value Heads ($h_{\text{KV}}$): $8$ (GQA factor $32/8 = 4\times$ reduction)
* Head Dimension ($d_{\text{head}}$): $128$
* Precision: BF16 ($2 \text{ bytes}$)
* Batch Size ($B$): $128$
* Sequence Length ($L_{\text{seq}}$): $8192 \text{ tokens}$

#### Calculation:
$$\text{Memory}_{\text{KVCache}} = 2 \times 128 \times 8192 \times 32 \times 8 \times 128 \times 2 \text{ bytes}$$

$$\text{Memory}_{\text{KVCache}} = 2 \times 128 \times 8192 \times 32 \times 1024 \times 2$$

$$\text{Memory}_{\text{KVCache}} = 137,438,953,472 \text{ bytes} \approx \mathbf{128 \text{ GB}}$$

If standard MHA had been used ($h_{\text{KV}} = 32$), this memory requirement would have quadrupled to **512 GB**. Modern structural variants like GQA are designed explicitly to alleviate this KV Cache memory wall.

---

## Summary Comparison: KV Cache Impact

| Dimension | Naive Generation (No Cache) | KV-Cached Generation |
| :--- | :--- | :--- |
| **Time Complexity per Token** | $O(t)$ | $O(1)$ Projections + $O(t)$ Attn |
| **Total Sequence Complexity** | $O(N^3)$ | $O(N^2)$ |
| **Primary System Bottleneck** | ALU Compute Bound (Excess FLOPs) | VRAM Memory Bandwidth Bound |
| **VRAM Memory Usage** | $O(1)$ (Only weights & current token) | $O(B \cdot L \cdot N_{\text{layers}} \cdot h_{\text{KV}} \cdot d_{\text{head}})$ |