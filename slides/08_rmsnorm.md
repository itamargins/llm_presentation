# RMSNorm Keeps Scale Invariance and Drops Everything Else

## Takeaway
Re-scaling is the ingredient that stabilizes training and re-centering is
not, so RMSNorm removes $\mu$ and $\beta$ and reduces the operator to a
single statistic.

## Purpose
Answers "what does the norm compute" after `07` answered "where does it
sit". Also supplies the first concrete implementation on screen, which
anchors the notation used in `12`.

## Content
- LayerNorm computes two statistics per token: mean $\mu$ and variance
  $\sigma^2$.
- RMSNorm computes one: the root mean square.
- What is dropped: mean-centering and the bias parameter $\beta$.
- Learnable parameters: $2 \times d_{\text{model}}$ ($\gamma, \beta$)
  reduced to $1 \times d_{\text{model}}$ ($\gamma$ only).
- Reported benefit: roughly 10–50% faster normalization step.
- The justification is empirical: scale invariance supplies the
  stability, mean-shift contributes negligibly.
- Used by Llama 3, Mistral, Qwen.

## Visual
Split layout.

- Left: the two equations stacked, with $\mu$ and $\beta$ visually
  struck through in the LayerNorm line. The strike-through is the slide's
  dominant visual signal.
- Right: `<div class="code-col">` with an ~8-line PyTorch `RMSNorm`
  forward, max 45 characters per line.
- Below the code: the 3-row comparison table (statistics computed,
  learnable parameters, speed).

## Equations
- $\text{LN}(x) = \dfrac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
  \odot \gamma + \beta$
- $\bar{a}_i = \dfrac{a_i}{\text{RMS}(a)} \odot \gamma_i$, where
  $\text{RMS}(a) = \sqrt{\dfrac{1}{d}\sum_{i=1}^{d} a_i^2 + \epsilon}$
- Definitions of $\mu$ and $\sigma^2$ only if the strike-through needs
  them to be legible.

## Technical constraints
- Present the mean-shift finding as an empirical observation, not a
  derived result.
- State the cost difference as **statistics computed** (two vs one), not
  as memory passes over the vector — fused LayerNorm kernels compute both
  statistics in a single pass, so the "2 passes" phrasing in the source
  table is an implementation-dependent claim.
- Keep $\epsilon$ inside the square root, consistent with the source.
- $\gamma$ is elementwise, not a matrix.
- Do not compare against BatchNorm and do not explain why normalization
  helps optimization in general.

## Source
`docs/02_Building_The_Complete_Transformer.md`, §3 (RMSNorm subsection
and the LayerNorm/RMSNorm comparison table).
