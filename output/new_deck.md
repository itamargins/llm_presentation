---
marp: true
paginate: true
math: katex
style: "@import '../design/theme.css';"
---

<style>
/* Compensates for design/theme.css: its base `section` rule is swallowed by
	 the YAML frontmatter lines at the top of that file and never applies.
	 Values below are copied verbatim from it, then adjusted for dense text. */
section {
	background-color: #ffffff;
	font-family: 'Helvetica Neue', Arial, sans-serif;
	color: #222222;
	padding: 50px 64px 50px 64px;
	font-size: 22px;
	line-height: 1.24;
}
/* Keep content-slide titles at a fixed top baseline. */
section:not(.section-header):not(.cover) {
	display: flex;
	flex-direction: column;
	justify-content: flex-start;
}
section h1,
section h2,
section h3,
section h4 {
	line-height: 1.16;
	margin-top: 0;
}
section h1 {
	font-size: 1.72em !important;
	margin-bottom: 0.35em;
}
section h2 {
	font-size: 1.38em !important;
	margin-bottom: 0.32em;
	padding-left: 0 !important;
}
section h3 {
	font-size: 1.12em !important;
	margin-bottom: 0.25em;
}
section h4 {
	font-size: 1.04em !important;
	margin-bottom: 0.24em;
}
/* Remove the orange corner bracket coming from theme h2 pseudo-elements. */
section h2::before,
section h2::after {
	content: none !important;
	display: none !important;
	width: 0 !important;
	height: 0 !important;
	background: none !important;
}
section ul {
	margin-top: 0.25em;
	margin-bottom: 0.2em;
}
section li {
	margin: 0.1em 0;
}
.columns {
	display: grid;
	grid-template-columns: 1fr 1fr;
	gap: 1.5rem;
	align-items: start;
}
section.section-header {
	display: flex;
	flex-direction: column;
	justify-content: center;
	text-align: center;
}
section.section-header h1,
section.section-header h2,
section.section-header h3 {
	margin: 0;
}
section.chapter-roadmap {
	justify-content: center;
}
section.chapter-roadmap h3 {
	font-size: 1.9em !important;
	margin-bottom: 0.5em;
}
section.chapter-roadmap ol {
	font-size: 1.45em;
	line-height: 1.4;
	margin: 0;
	padding-left: 1.25em;
}
section.chapter-roadmap li {
	margin: 0.2em 0;
}
section.spacious:not(.section-header):not(.cover) {
	font-size: 1.2em;
	line-height: 1.28;
}
section.spacious h2 {
	font-size: 1.66em !important;
	margin-bottom: 0.42em;
}
section.spacious h3 {
	font-size: 1.4em !important;
	margin-bottom: 0.34em;
}
section.spacious h4 {
	font-size: 1.22em !important;
	margin-bottom: 0.3em;
}
section.spacious p,
section.spacious li,
section.spacious blockquote {
	font-size: 1.1em;
}
section.spacious .fig img {
	max-height: 56vh;
}
section.cover {
	position: relative;
	text-align: center;
}
section.cover h1 {
	font-size: 2.65em !important;
	margin-top: 0.35em;
	margin-bottom: 0.4em;
}
section.cover .cover-subtitle {
	position: absolute;
	left: 64px;
	right: 64px;
	bottom: 22px;
	font-size: 0.86em;
	font-weight: 500;
	letter-spacing: 0.02em;
	color: #555555;
}
section.cover .fig img {
	max-height: 44vh;
}

/* Reserve ::after for page numbers; draw both accent bars on ::before. */
section::before {
	content: '';
	position: absolute;
	top: 0;
	right: 0;
	width: 50px;
	height: 100%;
	background: linear-gradient(to right, #E05315 0 12px, #5528CC 12px 50px);
	z-index: 10;
}
section::after {
	content: attr(data-marpit-pagination);
	position: absolute;
	right: 64px;
	bottom: 18px;
	top: auto;
	width: auto;
	height: auto;
	background: none !important;
	color: #222222;
	font-size: 24px;
	font-weight: 500;
	z-index: 20;
}
section::after-paginate {
	content: none !important;
}
.fig { text-align: center; }
.fig img { max-width: 100%; height: auto; }
.note { font-size: 0.6em; color: #555555; line-height: 1.35; }
table { font-size: 0.55em; }
</style>

<!-- _class: cover -->
# Anatomy of a Real-Life Transformer

<div class="fig">

![h:360](../assets/transformer.png)

</div>

<div class="cover-subtitle">MP Seminar - Sep-15 2026</div>

---

<!-- _class: chapter-roadmap -->

### Chapter Roadmap

1. Attention -> Transformer Block
2. Encoder-Decoder Architecture and Variants
3. Language Modeling and Generation
4. LLM Execution Phases
5. KV Cache
6. The Context Window ($C_{\text{max}}$)
7. Memory Optimization Techniques

---

<!-- _class: section-header -->
## 1. Attention -> Transformer Block

> In the last lecture, we saw how Self-Attention and Multi-Head Attention allow an embedded token to look at other tokens through a single, or multiple independent projection subspaces.
>
> How do we take those outputs, recombine them, push them through a deep network without destroying gradients, and scale this to an 8-billion parameter model that actually generates text?

---

### Output Linear Projection

<div class="fig">

![w:620](../assets/dot_product_mha.webp)

</div>

- Last time we ended with $h$ separate outputs of shape $[B, L, d_k]$ Simply concatenating MHA's $h$ outputs produces a tensor of shape $[B, L, h \cdot d_k] = [B, L, d_{\text{model}}]$.
- However, without a linear projection, information extracted by head $i$ remains trappsed and cannot interact with representations learned by head $j$.

So, a new linear projection (termed "output", $w_o$) is added.

---

### Output Projection $W^O$: Formula and Worked Example

<div class="columns">
<div>

The concatenated head representations are projected back into the residual space using the output projection matrix $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

- $\text{Concat}(\dots) \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$
- Output Tensor $\in \mathbb{R}^{B \times L \times d_{\text{model}}}$

</div>
<div>

Consider a Llama 3 8B configuration ($d_{\text{model}}=4096$, $h=32$, $d_k=128$, $B=2$, $L=512$):

1. Each head outputs $[2, 512, 128]$.
2. Concatenating across 32 heads yields $[2, 512, 4096]$.
3. Multiplying by $W^O \in \mathbb{R}^{4096 \times 4096}$ mixes feature dimensions across all heads, outputting $[2, 512, 4096]$ ready to be added back to the residual stream.

</div>
</div>

---

### Residual Connections Enable Direct Gradient Flow

<div class="columns">
<div>

In early deep networks, layers transformed representations sequentially ($x^{(l)} = f(x^{(l-1)})$), which causes vanishing/exploding gradients in ultra-deep (80+ layer) architectures.

The Transformer adopts a **residual stream view**:

$$x^{(l)} = x^{(l-1)} + \text{Attention}(x^{(l-1)}) + \text{FFN}(x^{(l-1)})$$

Unrolling over $L$ layers:

$$x^{(L)} = x^{(0)} + \sum_{l=1}^{L} \Delta x^{(l)}$$

</div>
<div>

During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to $x^{(0)}$ is:

$$\frac{\partial \mathcal{L}}{\partial x^{(0)}} = \frac{\partial \mathcal{L}}{\partial x^{(L)}} \left( I + \sum_{l=1}^{L} \frac{\partial \Delta x^{(l)}}{\partial x^{(0)}} \right)$$

The identity term $I$ guarantees that gradients flow backwards through all $L$ layers directly without decaying, eliminating the vanishing gradient wall regardless of depth.

</div>
</div>

---

### Normalization

#### Rationale: Stabilization of Activation Distributions

- As updates $\Delta x^{(l)}$ are repeatedly added to the residual stream, the variance of the hidden activations scales monotonically with depth ($Var(x^{(l)}) \approx Var(x^{(0)}) + \sum Var(\Delta x)$). Without normalization, deep activations explode, pushing Softmax inputs into saturated zero-gradient regimes.

- BatchNorm fails in sequence models because it relies heavily on batch size and suffers when dealing with variable sequence lengths and padding tokens-which corrupt cross-batch means and variances LayerNorm eliminates batch dependency entirely, providing identical, stable normalization behavior during both training and single-token inference without requiring inter-GPU synchronization during distributed training

- LayerNorm means normalizing per layer in the network. It is regular normalization over the activation outputs in the layer, and includes two additional learnable parameters: $\beta$ and $\gamma$, which are additional bias and scale terms, respectively:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

---

### RMSNorm (Root Mean Square Normalization)

Modern LLMs (Llama 3, Mistral, Qwen) replace standard LayerNorm with **RMSNorm** to improve computational speed without sacrificing variance stabilization.

Zhang and Sennrich (2019) demonstrated that the primary benefit of LayerNorm comes from scaling stability (rescaling the magnitude of inputs) rather than mean-shifting. Dropping mean-centering reduces memory access overhead and GPU latency while maintaining identical training stability.

Therefore, RMSNorm removes the mean-centering step and the bias parameter $\beta$:

$$\bar{a}_i = \frac{a_i}{\text{RMS}(a)} \odot \gamma_i, \quad \text{where } \text{RMS}(a) = \sqrt{\frac{1}{d} \sum_{i=1}^d a_i^2 + \epsilon}$$

Where:

- $a \in \mathbb{R}^{d_{\text{model}}}$ is the hidden activation vector for a single token.
- $\gamma \in \mathbb{R}^{d_{\text{model}}}$ is a learnable scaling parameter.
- $\epsilon$ is a small constant (e.g., $10^{-5}$) to prevent division by zero.

---

### Evolution: Post-LN vs. Pre-LN

- **Post-LayerNorm (Original Transformer):**

$$x^{(l)} = \text{LN}(x^{(l-1)} + f(x^{(l-1)}))$$

Gradients passing through the normalization operator in the main residual path are scaled inversely by the norm of the activations. Near the output layer, activations are large, making early layer updates tiny. This required delicate "warm-up" learning rate schedules.

- **Pre-LayerNorm (Modern Standard):**

$$x^{(l)} = x^{(l-1)} + f(\text{LN}(x^{(l-1)}))$$

Normalization is placed exclusively on the *branch* leading into the sub-layer. The main residual path remains pure identity, enabling stable zero-warmup training of 100B+ parameter models and larger learning rates to be used.

"the gradients are well-behaved without any exploding or vanishing for the Pre-LN Transformer both theoretically and empirically" (Xiong et al., 2020)

---

<!-- _class: spacious -->

### Post-LN vs. Pre-LN: Visual Comparison

<div class="fig">

![w:620](../assets/pre_post_ln.png)

</div>

<center>(a) - Post-LN, (b) - Pre-LN</center>

---

<!-- _class: spacious -->

### FFN

- The attention mixes information between tokens in the sequence, and the FFN is responsible for transforming information within the hidden dimension, for each token independently.
- It acts as "Fact Retrieval" - you can think of $w_1$ as a memory Key matrix ($d_{\text{model}} \times d_{\text{ff}}$) where each column represents a pattern detector, $w_2$ is the memory Value matrix, where each row represents a "concept" payload to be injected back into the residual stream.
- For example, during pre-training, when the model is repeatedly forced to predict "Paris" given "capital of France", gradient updates adjust the weights so that:
	- Column $i$ of $W_1$ aligns closely with the vector of "capital of France".
	- Row $i$ of $W_2$ aligns closely with the vector of "Paris".

---

### SwiGLU Gates Features Using a Swish-Activated Branch

<div class="columns">
<div>

GELU (Gaussian Error Linear Unit) and SiLU (Sigmoid Linear Unit) are "data-dependent dropout" (usually derived by using $x*P(x)$, Gaussian CDF and Sigmoid respectively).

<div class="fig">

![w:420](../assets/activations.png)

</div>

</div>
<div>

<div class="fig">

![w:420](../assets/swiglu.jpg)

</div>

- Two linear projections produce an up branch and a gate branch.
- The gate branch applies Swish and modulates the up branch via element-wise multiplication.
- A final down projection maps the gated representation back to $d_{\text{model}}$.

</div>
</div>

---

<!-- _class: spacious -->

### Parameter Matching Keeps SwiGLU Cost Equal to Standard FFN

Standard ReLU FFN uses 2 matrices ($W_{\text{gate/up}}, W_{\text{down}}$) with intermediate dimension $d_{\text{ff}} = 4 d_{\text{model}}$, yielding $2 \times 4 d_{\text{model}}^2 = 8 d_{\text{model}}^2$ parameters.

SwiGLU uses 3 weight matrices ($W_1, W_2, W_3$). To keep total parameters and FLOPs identical to a standard FFN, $d_{\text{ff}}$ is set to $\frac{8}{3} d_{\text{model}}$ (often rounded to the nearest multiple of 256 for GPU alignment):

$$3 \times \left( \frac{8}{3} d_{\text{model}} \times d_{\text{model}} \right) = 8 d_{\text{model}}^2 \text{ parameters}$$

In Llama 3 8B: $d_{\text{model}} = 4096 \implies d_{\text{ff}} = 14336 \approx \frac{8}{3} \times 4096$.

---

<!-- _class: spacious -->

### Complete Transformer Model = Embedding + $N$ Blocks + LM Head

$$x^{(l+1)} = x^{(l)} + \text{Attn}(\text{RMSNorm}(x^{(l)})) + \text{SwiGLU}(\text{RMSNorm}(x^{(l)}))$$

- A Transformer block is one residual update unit: normalization, sequence mixing, and feature transformation.
- A Transformer model stacks these blocks after token embedding and before the LM head logits projection.
- Depth lets contextual mixing and feature retrieval accumulate before next-token prediction.

---

<!-- _class: section-header -->
## 2. Encoder-Decoder Architecture and Variants

> Now we have a complete, fully functional Transformer block — with RMSNorm maintaining stability, Multi-Head Attention mixing sequence tokens, and SwiGLU FFN retrieving features.
>
> But a single block sitting on a whiteboard doesn't generate text, translate languages, or extract embeddings.
>
> To turn this block into an actual model, we have to make two fundamental engineering decisions: How do we stack them? and How do we restrict what each token is allowed to see?
>
> Historically, the field split into three distinct paths based on attention masking: models that look everywhere, models that look in two separate stages, and models that strictly look into the past. Let's look at the Encoder, the Decoder, and their hybrid friends.

---

### Encoder-Only vs. Decoder-Only: Attention Masking Sets the Use Case

<div class="columns">
<div>

**Encoder-Only (BERT)**

- Attention Mechanism: Unmasked, full Bi-Directional Self-Attention. Every token at position $i$ can attend to all other tokens at positions $j \in [1, L]$ (both past and future).
- Objective: Masked Language Modeling (MLM) or sequence classification.
- Primary Use Case: Producing rich contextual embeddings, feature extraction, passage reranking, and token classification.

</div>
<div>

**Decoder-Only (GPT/Llama)**

- Attention Mechanism: Causal (Masked) Self-Attention. Token $i$ can attend only to past and current positions $j \le i$.
- No Cross-Attention: Eliminates the Encoder-Decoder attention sub-layer entirely. Each block contains exactly two sub-layers (Causal MHA + FFN).
- Objective: Autoregressive Next-Token Prediction ($P(x_t \mid x_{<t})$).
- Primary Use Case: General-purpose LLMs, instruction following, and open-ended text generation.

</div>
</div>

---

### Encoder-Decoder

- The original paper suggested two sub-networks:
	- Encoder which processes the input sequence concurrently, producing keys and values (run once).
	- Decoder which generates the target sequence $Y_{tgt}$ autoregressively:
		- In the first iteration, it receives a "<START>" token ("shifted right") and the cached K and V from the encoder, to predict the next single token: T1.
		- In the next step, the decoder receives "<START> T1" (its own output) and the cached K,V from the encoder, to predict T2.
		- This goes on until the decoder itself predicts <END>.
	- The Encoder is essentially the "condition" upon which the decoder makes the prediction.

---

### Decoder-Only (GPT/Llama): Causal Masking

To enforce causality during parallel training and prefill processing, an upper-triangular masking matrix $M \in \mathbb{R}^{L \times L}$ sets future logit positions to $-\infty$ prior to the Softmax operation:

$$
M_{i,j} = \begin{cases}
0 & \text{if } i \ge j \\
-\infty & \text{if } i < j
\end{cases}
$$

$$
CausalAttention(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + M\right) V
$$

When $M_{i,j} = -\infty$, $\exp(-\infty) = 0$, completely blocking information leakage from future tokens $j > i$.

---

<!-- _class: section-header -->
## 3. Language Modeling and Generation

> How do we bridge the gap between this continuous hidden vector and a discrete token chosen from a vocabulary of 128,000 words?
>
> That is the job of the Language Modeling Head and the Decoding Pipeline.

---

<!-- _class: spacious -->

### LM Head (Unembedding)

After passing through all $L$ Transformer blocks, the final hidden state tensor $x^{(L)} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ represents the fully contextualized sequence representations.

To turn these representations into token predictions, the final vector at the last sequence position $t$, $x_t^{(L)} \in \mathbb{R}^{d_{\text{model}}}$, must be mapped to probability distributions over a discrete vocabulary $V$.

---

### LM Head Pipeline

<div class="fig">

![h:520](../assets/diagrams/lm_head_unembedding_pipeline.svg)

</div>

---

<!-- _class: spacious -->

### From Logits to Probabilities - Softmax & Temperature

Raw logits $z_t$ are scaled by a scalar **Temperature** parameter $T > 0$ before computing Softmax probabilities:

$$P(x_{t+1} = i \mid x_{1:t}) = \frac{\exp(z_{t, i} / T)}{\sum_{j=1}^{|V|} \exp(z_{t, j} / T)}$$

---

### Mathematical Behavior of Temperature $T$

- **$T \to 0$ (Greedy / Deterministic):**
	- The maximum logit dominates completely ($P(x_{t+1} = \arg\max z_t) \to 1$).
	- The model always picks the single most likely token, which leads to repetitive loops, factual rigidity, and generic responses.
- **$T = 1.0$ (Standard Softmax):**
	- Preserves the exact learned probability distribution of the model.
	- Samples from the full vocabulary long-tail. Extremely low-probability tokens (e.g., typos or rare words) will eventually be sampled, degrading coherence over long generations.
- **$T > 1.0$ (High Entropy / Creative):**
	- Flattens the logit landscape toward a uniform distribution ($P(x_{t+1}) \to \frac{1}{|V|}$), increasing sample variance and hallucination risk.

<div class="fig">

![w:520](../assets/diagrams/temperature_bars.svg)

</div>

---

### Temperature + Nucleus (Top-$p$) Sampling Mechanics

Nucleus Sampling dynamically adapts the candidate pool based on model confidence:

<div class="fig">

![h:360](../assets/diagrams/nucleus_sampling.svg)

</div>

Dynamic Behavior: When the model is uncertain (e.g., creative writing), probabilities are distributed broadly, so Nucleus sampling keeps many tokens ($k$ is large). When the model is very confident (e.g., factual queries), $P(i_1)$ alone might be $0.92$, so for $p=0.9$, $k=1$ and it automatically acts like greedy decoding.

---

<!-- _class: section-header -->
## 4. LLM Execution Phases

During inference, a Decoder-Only LLM executes in two distinct operational phases that exhibit radically different computational bottlenecks and hardware performance profiles.

---

<!-- _class: spacious -->

### Prefill

- Processing input prompt in parallel.
- This is done by Matrix-Matrix multiplications (GEMM) of shape $[B, L_{\text{prompt}}, d_{\text{model}}] \times [d_{\text{model}}, d_{\text{out}}]$.

---

<!-- _class: spacious -->

### Decode

- Generates output tokens serially, one step at a time ($L_{\text{step}} = 1$).
- The output token from step $t$ becomes the input token for step $t+1$.
- This is done by Matrix-Vector multiplications (GEMV) of shape $[B, 1, d_{\text{model}}] \times [d_{\text{model}}, d_{\text{out}}]$.

---

### Decode: Memory Background

- VRAM - main off-chip memory, on which model parameters and KV-cache are stored. Large (80 GB) but transfer is capped at 3TB/s.
- SRAM - Ultra-fast memory located directly next to the GPU. Operates at ~10x speed of VRAM, but very small (~50MB total).
- For every single token generated, the GPU must fetch all model weights (e.g., 14-16 GB for an 8B FP16 model) from VRAM into fast SRAM caches.
- This means the GPU cores are mostly idle, only to perform a tiny number of calculations on a single vector.

---

<!-- _class: section-header -->
## 5. KV Cache

> Now, ask yourself a critical question: What happens to the Attention calculation when we generate token $N+1$?
>
> To compute Attention for that new token, it needs to attend to Query $N+1$ against the Keys and Values of every single preceding token in the sequence—tokens 1 all the way through $N$.
>
> If we naively recompute the Query, Key, and Value vectors for all $N$ past tokens at every step, our sequence computation scales quadratically at $O(L^2)$. By the time you reach token 2,000, your GPU is spending 99% of its time recalculating math it already did 1,999 steps ago.
>
> To fix this, we introduce the single most critical data structure in modern LLM inference: The Key-Value (KV) Cache. Instead of recomputing the past, we save the $K$ and $V$ tensors generated during the Prefill phase, store them in GPU memory, and simply concatenate the new token's Key and Value at every decode step. We trade memory capacity to buy back computational speed.

---

<!-- _class: spacious -->

### Useful References for Chapter 5

- https://www.youtube.com/watch?v=7OrMFn86PlM
- https://www.youtube.com/watch?v=RUlQmkFY4F8
- https://www.youtube.com/watch?v=gpp57x_z_Jg

- Explain the problem.
- How does KV cache help.

---

### KV Cache: Formula and Variables

$$\text{KV Cache Bytes} = 2 \times b \times s \times l \times h_{kv} \times d_h \times P$$

Where:

- $2$: Accounts for storing both Key ($K$) and Value ($V$) matrices.
- $b$: Batch size (number of parallel concurrent requests).
- $s$: Sequence length (total tokens in context + generation).
- $l$: Number of hidden layers.
- $h_{kv}$: Number of Key/Value heads per layer.
- $d_h$: Dimension per head ($d_{\text{head}}$).
- $P$: Precision size in bytes ($P = 2$ bytes for FP16/BF16, $P = 1$ byte for INT8/FP8).

---

### KV Cache: Reference Architecture (Llama-3 70B)

- Layers ($l$): 80
- Query Heads ($h_q$): 64
- Head Dimension ($d_h$): 128
- Precision ($P$): 2 bytes (BF16)
- Context Length ($s$): 8,192 tokens
- Batch Size ($b$): 16 concurrent sequences

- Speed improvement.
- What does it cost?

---

<!-- _class: section-header -->
## 6. The Context Window ($C_{\text{max}}$)

> Every model provider advertises massive context lengths today—32k, 128k, 1M, or even 2M tokens. But context length isn't just a setting you toggle in a config file. It is a strict physical boundary bounded by three distinct engineering brick walls:
>
> First, The Compute Wall: $O(L^2)$ quadratic complexity in the attention matrix during prompt ingestion.
>
> Second, The Memory Wall: The exponential VRAM footprint of the KV Cache during token generation—the exact bottleneck we calculated in Chapter 5.
>
> And third, The Positional Representation Wall: The mathematical failure of Rotary Position Embeddings (RoPE) when pushed outside their training distribution.

---

<!-- _class: section-header -->
## 7. Memory Optimization Techniques

---

<!-- _class: spacious -->

### Useful References for Chapter 7

- Deep dives on KV-cache and memory-aware attention kernels.

- https://www.youtube.com/watch?v=o68RRGxAtDo
- https://www.youtube.com/watch?v=_pWigIleZNs

---

### Multi-Query Attention

<div class="fig">

![w:760](../assets/mqa.jpg)

</div>

---

### Grouped-Query Attention

<div class="fig">

![w:760](../assets/gqa.jpg)

</div>

---

### MHA vs. MQA vs. GQA

<div class="fig">

![w:760](../assets/attn_sbs.jpg)

</div>

---

<!-- _class: spacious -->

### Training Note (MQA/GQA)

- Two linear projections produce an "up" branch and a "gate" branch.
- The gate branch applies Swish and modulates the up branch via element-wise multiplication.
- A final down projection maps the gated representation back to $d_{\text{model}}$.

---

<!-- _class: spacious -->

### DeepSeek improvement??? (https://www.youtube.com/watch?v=9y-0rpEnPrg) - MLA? (Latent Attention)

- MLA as a low-rank compression technique (compressing $K$ and $V$ into a tiny latent vector $c_t^{KV}$ via joint projection), showing how state-of-the-art architectures compress the cache even further without sacrificing MHA-level representation power.

---

### Paged Attention?

- Problem: Traditional KV cache allocation requires contiguous memory blocks in VRAM per request, leading to severe virtual memory fragmentation (up to 60-80% memory waste).
- Solution: Operates like virtual memory paging in operating systems. It splits the KV Cache into fixed-size physical memory blocks (e.g., 16 tokens per block) allocated dynamically via a lookup page table.
- Impact: Eliminates memory fragmentation, enabling significantly higher batch sizes on fixed GPU hardware.

---

### Flash Attention?

Reference: https://www.youtube.com/watch?v=RcFrRqcV4ZA

- The Problem: Standard attention computes $A = \text{softmax}(QK^T / \sqrt{d})V$. Storing that intermediate $N \times N$ attention matrix $A$ in High Bandwidth Memory (HBM) creates an $O(N^2)$ memory footprint and makes attention memory-bandwidth bound rather than compute-bound.
- The Solution: FlashAttention uses tiling (online softmax) to compute attention block-by-block inside fast SRAM on the GPU chip without ever writing the massive $N \times N$ matrix back to HBM.
- The Takeaway: While GQA saves VRAM space, FlashAttention gives you raw wall-clock speedup and exact (non-approximated) attention computation.

<div class="fig">

![w:760](../assets/llm_blocks.png)

</div>

---

<!-- _class: section-header -->
## More Cool Stuff

---

<!-- _class: spacious -->

### https://hfviewer.com/ - architectures and glossary

- Any model on HuggingFace can be visualized - simply replace the url.
	- Transformer Block: https://hfviewer.com/glossary/transformer-block/
	- GQA: https://hfviewer.com/glossary/grouped-query-attention/
	- Qwen3.8-27B: https://hfviewer.com/Qwen/Qwen3.8-27B

---

<!-- _class: spacious -->

### End-to-End Takeaway: Architecture Meets Hardware Constraints

- We built a modern decoder block from RMSNorm, residual paths, attention, and SwiGLU.
- Stacking blocks gives a decoder-only LM: hidden states become logits, then sampled tokens.
- Inference has two distinct phases: Prefill is compute-bound; Decode is memory-bandwidth-bound.
- The practical bottleneck is not only FLOPs, but also memory movement and cache layout.

---

<!-- _class: spacious -->

### Memory-Aware Kernels Turn the $O(L^2)$ Wall into Throughput

- GQA reduces KV memory by sharing keys/values across query groups.
- MLA compresses KV information into low-rank latent states.
- FlashAttention tiles computation in SRAM and avoids writing full $N \times N$ attention maps to HBM.
- PagedAttention removes VRAM fragmentation through fixed-size block paging.
- Together, these changes make long-context inference production-feasible.

---

<!-- _class: spacious -->

### References

SwiGLU:
	https://www.youtube.com/watch?v=2FaI2Fen1mQ
	https://www.youtube.com/watch?v=CXqx5LDOfs4

General LLM's:
	https://www.youtube.com/watch?v=BprirYymXrg
