---
marp: true
paginate: true
style: "@import '../design/theme.css';"
---

<style>
/* Compensates for design/theme.css: its base `section` rule is swallowed by
   the YAML frontmatter lines at the top of that file and never applies.
   Values below are copied verbatim from it. Delete this rule once the
   stylesheet is fixed at the source. */
section {
  background-color: #ffffff;
  font-family: 'Helvetica Neue', Arial, sans-serif;
  color: #222222;
  padding: 60px 80px 60px 80px;
  font-size: 26px;
}
.columns {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1.5rem;
  align-items: start;
}
.code-col pre {
  font-size: 62% !important;
  white-space: pre-wrap !important;
}
.fig { text-align: center; }
.fig img { max-width: 100%; height: auto; }
.note { font-size: 0.6em; color: #555555; line-height: 1.35; }
table { font-size: 0.55em; }
</style>

## A Transformer Layer Is Two Sub-Layers Writing Deltas onto One Residual Stream

<div class="fig">

![h:430](../assets/diagrams/block_topology.svg)

</div>

---

## Concatenating Heads Leaves Them Isolated; $W^O$ Is Where Subspaces Mix

<div class="columns">

<div>

$$\text{head}_i = \text{Softmax}\!\left(\frac{Q_i K_i^T}{\sqrt{d_k}} + M\right) V_i$$

$$\text{MHA} = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O$$

$$W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$$

<p class="note">
Concat is already d_model wide. W<sup>O</sup> is not a reshape — it is the
only operator that mixes head subspaces.
</p>

</div>

<div class="fig">

![w:520](../assets/diagrams/head_mixing.svg)

</div>

</div>

---

## Sub-Layers Read the Residual Stream and Write Deltas; the Identity Path Is Never Touched

<div class="columns">

<div class="fig">

![w:520](../assets/diagrams/residual_bus.svg)

</div>

<div>

$$x^{(L)} = x^{(0)} + \sum_{l=1}^{L} \Delta x^{(l)}$$

$$\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x^{(0)}}
&= \frac{\partial \mathcal{L}}{\partial x^{(L)}}
   \frac{\partial x^{(L)}}{\partial x^{(0)}} \\[2pt]
&= \frac{\partial \mathcal{L}}{\partial x^{(L)}}
   \left( I + \sum_{l=1}^{L}
   \frac{\partial \Delta x^{(l)}}{\partial x^{(0)}} \right)
\end{aligned}$$

<p class="note">
The identity term does not depend on L: at 80+ layers, no layer has to
re-learn an identity mapping.
</p>

</div>

</div>

---

## Norm Placement Removes Warm-Up: Pre-LN Leaves the Identity Path Exact

$$\text{Var}(x^{(l)}) \approx \text{Var}(x^{(0)}) + \textstyle\sum \text{Var}(\Delta x) \;\Longrightarrow\; \text{a norm is required}$$

<div class="columns">

<div class="fig">

![w:490](../assets/diagrams/post_ln.svg)

$$x^{(l)} = \text{LN}\!\left(x^{(l-1)} + f(x^{(l-1)})\right)$$

<p class="note">
The residual path runs <b>through</b> the norm. Gradients are scaled
inversely by the activation norm — large near the output, so early layers
barely update. Requires warm-up.
</p>

</div>

<div class="fig">

![w:490](../assets/diagrams/pre_ln.svg)

$$x^{(l)} = x^{(l-1)} + f\!\left(\text{LN}(x^{(l-1)})\right)$$

<p class="note">
The norm sits on the branch only. The residual path stays exactly the
identity — stable zero-warmup training at 100B+ parameters.
</p>

</div>

</div>

---

## RMSNorm Keeps Scale Invariance and Drops Everything Else

<div class="columns">

<div>

$$\text{LN}(x) = \frac{x - \cancel{\mu}}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \cancel{\beta}$$

$$\bar{a}_i = \frac{a_i}{\text{RMS}(a)} \odot \gamma_i$$

$$\text{RMS}(a) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} a_i^2 + \epsilon}$$

<p class="note">
An empirical observation, not a derivation: scale invariance supplies the
stability, the mean shift contributes negligibly.<br/>
Used by Llama 3, Mistral, Qwen.
</p>

</div>

<div class="code-col">

```python
def rms_norm(x, g, eps=1e-5):
    # one statistic: no mean, no beta
    ms = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(ms + eps)
    return x * g          # g: [d_model]
```

| | LayerNorm | RMSNorm |
| :--- | :--- | :--- |
| Statistics | $\mu$ and $\sigma^2$ | RMS only |
| Parameters | $2 d_{\text{model}}$ | $d_{\text{model}}$ |
| Speed | baseline | ~10–50% faster |

</div>

</div>

---

## Attention Routes Across Tokens; the FFN Is a Per-Token Associative Memory

<div class="columns">

<div class="fig">

![w:500](../assets/diagrams/routing_grid.svg)

</div>

<div class="fig">

![w:520](../assets/diagrams/ffn_memory_unit.svg)

</div>

</div>

$$\begin{aligned}
\text{FFN}(x) &= \sigma(x W_1 + b_1) W_2 + b_2 \\
&= \sum_{k=1}^{d_{\text{ff}}} \sigma(x \cdot w_{1,k} + b_{1,k})\, w_{2,k} \; + \; b_2
\end{aligned}$$

---

## SwiGLU Buys a Multiplicative Gate and Pays for It by Shrinking $d_{\text{ff}}$ to $\tfrac{8}{3} d_{\text{model}}$

<div class="columns">

<div>

$$\text{SwiGLU}(x) = \left( \text{SiLU}(x W_1) \otimes x W_2 \right) W_3$$

$$\text{SiLU}(x) = x \cdot \sigma(x)$$

$$3 \times \left( \tfrac{8}{3} d_{\text{model}} \times d_{\text{model}} \right) = 8\, d_{\text{model}}^2$$

<p class="note">
The same budget as a 2-matrix FFN at d_ff = 4·d_model.<br/>
d_ff is rounded to a hardware-aligned multiple (e.g. of 256).
</p>

</div>

<div class="fig">

![w:480](../assets/diagrams/swiglu_dataflow.svg)

</div>

</div>

---

## The LM Head Closes the Loop: Final Norm, Unembedding, Temperature-Scaled Softmax

<div class="columns">

<div class="fig">

![w:320](../assets/diagrams/lm_head_pipeline.svg)

$$z_t = \text{RMSNorm}\!\left(x_t^{(L)}\right) W_{\text{vocab}}$$

</div>

<div class="fig">

![w:520](../assets/diagrams/temperature_bars.svg)

$$P(x_{t+1} = i \mid x_{1:t}) = \frac{\exp(z_{t,i}/T)}{\sum_{j} \exp(z_{t,j}/T)}$$

</div>

</div>

---

## One Block in Six Equations: The Complete Llama-3-Style Forward Pass

<div class="columns">

<div>

$$\begin{aligned}
(1)\;\; \tilde{x}^{(l-1)} &= \text{RMSNorm}(x^{(l-1)}) \\[1pt]
(2)\;\; a^{(l)} &= \text{GQA}(\tilde{x}^{(l-1)})\, W^O \\[1pt]
(3)\;\; x^{(l-1/2)} &= x^{(l-1)} + a^{(l)} \\[1pt]
(4)\;\; \tilde{x}^{(l-1/2)} &= \text{RMSNorm}(x^{(l-1/2)}) \\[1pt]
(5)\;\; f^{(l)} &= \left( \text{SiLU}(\tilde{x}^{(l-1/2)} W_1) \right. \\
&\qquad \left. \otimes\; (\tilde{x}^{(l-1/2)} W_2) \right) W_3 \\[1pt]
(6)\;\; x^{(l)} &= x^{(l-1/2)} + f^{(l)}
\end{aligned}$$

</div>

<div class="code-col">

```python
def block(x):
    xn = rms_norm(x, g1)      # (1)
    a  = attn_gqa(xn) @ W_O   # (2)
    x  = x + a                # (3)
    xn = rms_norm(x, g2)      # (4)
    f  = (silu(xn @ W1)       # (5)
          * (xn @ W2)) @ W3
    return x + f              # (6)
```

<p class="note">
Chapter III: what changes when steps (1)–(6) are run once per generated
token.
</p>

</div>

</div>
