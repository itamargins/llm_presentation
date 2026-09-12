# II. Building the Complete Transformer Block: Functional Rationale

Moving beyond the core attention operation ($QK^T / \sqrt{d_k} \cdot V$), a raw scaled dot-product matrix on its own cannot form a deep network, store factual knowledge, or maintain numerical stability across 80+ stacked layers. 

A modern Transformer layer (using Llama 3 as our baseline reference architecture) is an engineered pipeline designed around two primary operations: **routing information across sequence elements (Attention)** and **processing/storing information within sequence elements (Feed-Forward Network / MLP)**.

              Input Tensor x^(l-1)  [B, L, d_model]
                       │
                       ├──┐
                       │  ▼
                       │  RMSNorm
                       │  │
                       │  ▼
                       │  Multi-Head / Grouped-Query Attention
                       │  │
                       │  ▼
                       │  Linear Projection (W^O)
                       │  │
                       ▼  ▼
                       ┌──┴──┐
                       │  +  │  <-- Residual Addition (Attention Delta)
                       └──┬──┘
                          │
                          ├──┐
                          │  ▼
                          │  RMSNorm
                          │  │
                          │  ▼
                          │  SwiGLU Feed-Forward Network (FFN)
                          │  │
                          ▼  ▼
                          ┌──┴──┐
                          │  +  │  <-- Residual Addition (FFN Delta)
                          └──┬──┘
                             │
                             ▼
              Output Tensor x^(l)   [B, L, d_model]


---

## 1. The Multi-Head Output Projection ($W^O$)

### Functional Rationale & Problem
Multi-Head Attention (MHA) splits the hidden dimension $d_{\text{model}}$ into $h$ independent attention heads, each operating in a subspace of dimension $d_k = d_{\text{model}} / h$. 

Each head $i$ independently computes a context-weighted value matrix:

$$\text{head}_i = \text{Softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i \in \mathbb{R}^{B \times L \times d_k}$$

Simply concatenating these $h$ outputs produces a tensor of shape $[B, L, h \cdot d_k] = [B, L, d_{\text{model}}]$. However, without a linear projection:
1. **Head Isolation:** Information extracted by head $i$ remains trapped in slice $[i \cdot d_k : (i+1) \cdot d_k]$ and cannot interact with representations learned by head $j$.
2. **Subspace Mixing Deficit:** The model cannot linearly recombine features extracted across different attention perspectives (e.g., combining a syntactic dependency head with a long-range semantic reference head).

### Mathematical Formulation
The concatenated head representations are projected back into the residual space using the output projection matrix $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$:

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$$

Where:
* $\text{Concat}(\dots) \in \mathbb{R}^{B \times L \times d_{\text{model}}}$
* $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$
* Output Tensor $\in \mathbb{R}^{B \times L \times d_{\text{model}}}$

### Concrete Tensor Example
Consider a Llama 3 8B configuration:
* Hidden dimension $d_{\text{model}} = 4096$
* Query heads $h = 32$
* Per-head dimension $d_k = 4096 / 32 = 128$
* Batch size $B = 2$, Sequence length $L = 512$

1. Each head outputs $[2, 512, 128]$.
2. Concatenating across 32 heads yields $[2, 512, 32 \times 128] = [2, 512, 4096]$.
3. Multiplying by $W^O \in \mathbb{R}^{4096 \times 4096}$ mixes feature dimensions across all heads, outputting a tensor of $[2, 512, 4096]$ ready to be added back to the residual stream.

---

## 2. The Residual Stream ("The Shared Bus")

### Functional Rationale & Mechanics
In early deep networks, layers transformed representations sequentially ($x^{(l)} = f(x^{(l-1)})$). In ultra-deep architectures (80+ layers), this creates severe vanishing/exploding gradient problems and forces each layer to re-learn identity mappings if no transformation is needed.

The Transformer adopts a **residual stream view**:

$$x^{(l)} = x^{(l-1)} + \text{Attention}(x^{(l-1)}) + \text{FFN}(x^{(l-1)})$$

Instead of transforming the representation directly, each sub-layer (Attention or FFN) acts as a **read/write head on a shared memory bus**:
* **Read:** The sub-layer reads the current hidden state $x^{(l-1)}$ from the bus.
* **Compute:** It calculates a sparse update/delta vector ($\Delta x$).
* **Write:** It adds $\Delta x$ back onto the bus via vector addition.

Residual Stream (Bus)  =========================================================>
│                                      ▲
│ (Read)                               │ (Write +)
▼                                      │
[ Sub-Layer Block ] ─────────── (Delta Δx) ────┘


### Mathematical & Gradient Impact
Unrolling the recursion over $L$ layers yields:

$$x^{(L)} = x^{(0)} + \sum_{l=1}^{L} \Delta x^{(l)}$$

During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to the input state $x^{(0)}$ is:

$$\frac{\partial \mathcal{L}}{\partial x^{(0)}} = \frac{\partial \mathcal{L}}{\partial x^{(L)}} \frac{\partial x^{(L)}}{\partial x^{(0)}} = \frac{\partial \mathcal{L}}{\partial x^{(L)}} \left( I + \sum_{l=1}^{L} \frac{\partial \Delta x^{(l)}}{\partial x^{(0)}} \right)$$

The identity term $I$ guarantees that gradients flow backwards through all $L$ layers directly without decaying, eliminating the vanishing gradient wall regardless of depth.

---

## 3. Normalization (Pre-LayerNorm vs. Post-LayerNorm & RMSNorm)

### Rationale: Stabilization of Activation Distributions
As updates $\Delta x^{(l)}$ are repeatedly added to the residual stream, the variance of the hidden activations scales monotonically with depth ($Var(x^{(l)}) \approx Var(x^{(0)}) + \sum Var(\Delta x)$). Without normalization, deep activations explode, pushing Softmax inputs into saturated zero-gradient regimes.

### Evolution: Post-LN vs. Pre-LN
* **Post-LayerNorm (Original Transformer):** 
  $$x^{(l)} = \text{LN}(x^{(l-1)} + f(x^{(l-1)}))$$
  Gradients passing through the normalization operator in the main residual path are scaled inversely by the norm of the activations. Near the output layer, activations are large, making early layer updates tiny. This required delicate "warm-up" learning rate schedules.
* **Pre-LayerNorm (Modern Standard):**
  $$x^{(l)} = x^{(l-1)} + f(\text{LN}(x^{(l-1)}))$$
  Normalization is placed exclusively on the *branch* leading into the sub-layer. The main residual path remains pure identity, enabling stable zero-warmup training of 100B+ parameter models.

Post-LN:  x --[Sub-Layer]--> (+) --[LayerNorm]--> Output
│                   ▲
└───────────────────┘

Pre-LN:   x ──┬──[LayerNorm]──[Sub-Layer]──┐
│                            ▼
└───────────────────────────(+) ───────> Output


### RMSNorm (Root Mean Square Normalization)
Modern LLMs (Llama 3, Mistral, Qwen) replace standard LayerNorm with **RMSNorm** to improve computational speed without sacrificing variance stabilization.

Standard LayerNorm computes both mean $\mu$ and variance $\sigma^2$:

$$\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta, \quad \text{where } \mu = \frac{1}{d}\sum_{i=1}^d x_i, \; \sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$$

RMSNorm makes an empirical observation: **the scaling invariance of LayerNorm provides stability, while mean-shifting ($\mu$) provides negligible benefit.** 

By assuming a mean of zero, RMSNorm removes the mean-centering step and the bias parameter $\beta$:

$$\bar{a}_i = \frac{a_i}{\text{RMS}(a)} \odot \gamma_i, \quad \text{where } \text{RMS}(a) = \sqrt{\frac{1}{d} \sum_{i=1}^d a_i^2 + \epsilon}$$

Where:
* $a \in \mathbb{R}^{d_{\text{model}}}$ is the hidden activation vector for a single token.
* $\gamma \in \mathbb{R}^{d_{\text{model}}}$ is a learnable scaling parameter.
* $\epsilon$ is a small constant (e.g., $10^{-5}$) to prevent division by zero.

### Mathematical Comparison & Computational Advantage
| Metric | LayerNorm | RMSNorm |
| :--- | :--- | :--- |
| **Operations per Token** | 2 passes over vector (Mean, then Var) | 1 pass over vector (Sum of squares) |
| **Learnable Parameters** | $2 \times d_{\text{model}}$ ($\gamma, \beta$) | $1 \times d_{\text{model}}$ ($\gamma$ only) |
| **Speed Benefit** | Baseline | ~10–50% faster normalization step |

---

## 4. The Feed-Forward Network (FFN / SwiGLU)

### Functional Rationale: Spatial Routing vs. Content Processing
A crucial distinction in Transformer design:
* **Attention Sub-Layer:** Operates *across space/time* (across sequence length $L$). It **routes** and **gathers** information between different token vectors, but performs minimal feature transformation within a token vector.
* **FFN Sub-Layer:** Operates *strictly within space/time* (independently per token index $t$). It **processes** and **transforms** features within a single token vector across channel dimensions.

Attention:  [Token 1] <───> [Token 2] <───> [Token 3]  (Cross-Token Information Routing)
│               │               │
FFN:        [   MLP   ]     [   MLP   ]     [   MLP   ]  (Per-Token Feature Transformation)


### The Associative Key-Value Memory Interpretation
The FFN expands the hidden dimension $d_{\text{model}}$ to an intermediate dimension $d_{\text{ff}}$ (typically $4 \times d_{\text{model}}$ or $\frac{8}{3} \times d_{\text{model}}$) before projecting back down.

Mathematically, a standard two-layer FFN with activation $\sigma$:

$$\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2 = \sum_{k=1}^{d_{\text{ff}}} \sigma(x \cdot w_{1,k} + b_{1,k}) w_{2,k}$$

This can be interpreted as an **associative key-value memory**:
* **Keys ($W_1$):** Each row $w_{1,k}$ acts as a concept detector pattern key. The dot product $x \cdot w_{1,k}$ detects whether concept $k$ exists in the token's current representation.
* **Values ($W_2$):** If concept $k$ fires ($\sigma(\cdot) > 0$), vector $w_{2,k}$ is written to the token's residual state, inserting factual knowledge, syntactic role updates, or semantic attributes.

### Evolution to SwiGLU (Swish Gated Linear Unit)
Standard architectures used ReLU or GELU activations. Contemporary SOTA models use **SwiGLU** (PaLM, Llama 3).

A Gated Linear Unit (GLU) multiplies two linear transformations element-wise, where one branch acts as a continuous non-linear gate:

$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_1) \otimes x W_2 \right) W_3$$

Where:
* $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ (Gate Projection)
* $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ (Up Projection)
* $W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ (Down Projection)
* $\otimes$ denotes element-wise (Hadamard) product.
* $\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$ (also known as Swish-1).

```mermaid
                  Input x [d_model]
                          │
           ┌──────────────┴──────────────┐
           ▼                             ▼
     Linear (W_1)                  Linear (W_2)
           │                             │
           ▼                             │
     SiLU Activation                     │
           │                             │
           └──────────────┬──────────────┘
                          ▼
                 Element-wise Mult (⊗)
                          │  [d_ff]
                          ▼
                    Linear (W_3)
                          │
                          ▼
               Output Delta [d_model]
```

### Parameter-Matching Strategy for SwiGLU
Standard ReLU FFN uses 2 matrices ($W_{\text{gate/up}}, W_{\text{down}}$) with intermediate dimension $d_{\text{ff}} = 4 d_{\text{model}}$, yielding $2 \times 4 d_{\text{model}}^2 = 8 d_{\text{model}}^2$ parameters.

SwiGLU uses 3 weight matrices ($W_1, W_2, W_3$). To keep total parameters and FLOPs identical to a standard FFN, $d_{\text{ff}}$ is set to $\frac{8}{3} d_{\text{model}}$ (often rounded to the nearest multiple of 256 for GPU alignment):

$$3 \times \left( \frac{8}{3} d_{\text{model}} \times d_{\text{model}} \right) = 8 d_{\text{model}}^2 \text{ parameters}$$

In Llama 3 8B: $d_{\text{model}} = 4096 \implies d_{\text{ff}} = 14336 \approx \frac{8}{3} \times 4096$.

---

## 5. The Language Modeling Head (LM Head) & Sampling Loop

After passing through all $L$ Transformer blocks, the final hidden state tensor $x^{(L)} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ represents the fully contextualized sequence representations.

To turn these representations into token predictions, the final vector at the last sequence position $t$, $x_t^{(L)} \in \mathbb{R}^{d_{\text{model}}}$, must be mapped to probability distributions over a discrete vocabulary $V$.

Final Hidden Vector x_t^{(L)}  [d_model]
│
▼
RMSNorm
│
▼
Linear LM Head (W_vocab)  [d_model -> |V|]
│
▼
Raw Logits z_t        [|V|]
│
▼
Temperature Scaling (z_t / T)
│
▼
Softmax Normalization
│
▼
Next-Token Probabilities P(x_{t+1})


### 1. Unembedding Projection
The final hidden state is normalized and projected onto the vocabulary dimension using unembedding matrix $W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$:

$$z_t = \text{RMSNorm}(x_t^{(L)}) W_{\text{vocab}} \in \mathbb{R}^{|V|}$$

Where $z_t$ is a vector of unnormalized log-probabilities (**logits**) for every token in vocabulary $V$.

*Note on Weight Tying:* In some architectures, $W_{\text{vocab}}$ shares weights with the input token embedding matrix ($W_{\text{embed}}^T$). Modern large models (e.g., Llama 3) often un-tie these matrices to allow $W_{\text{vocab}}$ to optimize output logits independently of input representations.

### 2. Softmax & Temperature Scaling
Raw logits $z_t$ are scaled by a scalar **Temperature** parameter $T > 0$ before computing Softmax probabilities:

$$P(x_{t+1} = i \mid x_{1:t}) = \frac{\exp(z_{t, i} / T)}{\sum_{j=1}^{|V|} \exp(z_{t, j} / T)}$$

#### Mathematical Behavior of Temperature $T$:
* **$T \to 0$ (Greedy / Deterministic):** The maximum logit dominates completely ($P(x_{t+1} = \arg\max z_t) \to 1$). The model always picks the single most likely token.
* **$T = 1.0$ (Standard Softmax):** Preserves the exact learned probability distribution of the model.
* **$T > 1.0$ (High Entropy / Creative):** Flattens the logit landscape toward a uniform distribution ($P(x_{t+1}) \to \frac{1}{|V|}$), increasing sample variance and hallucination risk.

---

## Comprehensive Mathematical Summary of a Full Block (Llama 3 Style)

Given input hidden tensor $x^{(l-1)} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ into layer $l$:

$$\text{Step 1: Attention Pre-Norm} \quad \tilde{x}^{(l-1)} = \text{RMSNorm}(x^{(l-1)})$$

$$\text{Step 2: Attention \& Projection} \quad a^{(l)} = \text{GQA}(\tilde{x}^{(l-1)}) W^O$$

$$\text{Step 3: Attention Residual Write} \quad x^{(l-1/2)} = x^{(l-1)} + a^{(l)}$$

$$\text{Step 4: FFN Pre-Norm} \quad \tilde{x}^{(l-1/2)} = \text{RMSNorm}(x^{(l-1/2)})$$

$$\text{Step 5: SwiGLU FFN} \quad f^{(l)} = \left( \text{SiLU}(\tilde{x}^{(l-1/2)} W_1) \otimes (\tilde{x}^{(l-1/2)} W_2) \right) W_3$$

$$\text{Step 6: FFN Residual Write} \quad x^{(l)} = x^{(l-1/2)} + f^{(l)}$$