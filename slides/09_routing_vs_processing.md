# Attention Routes Across Tokens; the FFN Is a Per-Token Associative Memory

## Takeaway
The FFN never mixes positions — it is a key-value lookup executed
independently at every token, and it is where per-token knowledge is
written into the residual stream.

## Purpose
Delivers the chapter's central functional distinction, announced on `04`
and only now justified. Without it, the FFN looks like generic extra
capacity rather than the component that answers "where are facts stored".

## Content
- Axis of operation: attention operates across the sequence axis $L$;
  the FFN operates strictly across the channel axis $d$, independently
  per token index $t$, with weights shared across positions.
- Attention routes and gathers between token vectors but performs
  minimal feature transformation within one.
- The FFN expands $d_{\text{model}} \to d_{\text{ff}} \to
  d_{\text{model}}$ ($d_{\text{ff}}$ typically $4 d_{\text{model}}$;
  the gated variant's ratio belongs to `10`).
- Associative key-value memory reading, per intermediate unit $k$:
  the dot product with the key vector detects whether concept $k$ is
  present; if it fires, the corresponding value vector is written to the
  token's residual state.
- This is what makes the summation form, not the matrix form, the
  equation that earns space on the slide.

## Visual
Full-width, two panels.

- Panel 1: a token grid. Attention drawn as horizontal edges between
  columns; the FFN drawn as identical vertical stamps, one per column.
  The dominant signal is the **absence of horizontal edges** in the FFN
  row.
- Panel 2: a single intermediate unit $k$ firing — key match on the input
  vector, then its value vector added onto that one token's residual
  slice.
- Reuse the residual-bus styling from `06` in panel 2 so the write is
  recognizably the same operation.

## Equations
- $\text{FFN}(x) = \sigma(x W_1 + b_1) W_2 + b_2
  = \sum_{k=1}^{d_{\text{ff}}} \sigma(x \cdot w_{1,k} + b_{1,k})\,
  w_{2,k} \; (+\, b_2)$

## Technical constraints
- **Index orientation correction.** With $x W_1$ and
  $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, the
  detector $w_{1,k}$ is a **column** of $W_1$ and $w_{2,k}$ is a **row**
  of $W_2$. The source calls $w_{1,k}$ a row, which is wrong for this
  convention.
- The summation form omits $b_2$. Either carry $+\, b_2$ explicitly or
  mark the decomposition as holding up to the output bias; do not print
  it as a bare equality.
- The FFN stamps must be visually identical across columns — shared
  weights, not per-position parameters.
- No horizontal connection may appear anywhere in the FFN panel.
- Keep the memory reading at the level the source supports (concept
  detector / written value). Do not import interpretability results,
  neuron-naming claims, or capacity estimates from outside the document.
- Do not explain what an MLP is, why non-linearity is required, or
  universal approximation.

## Source
`docs/02_Building_The_Complete_Transformer.md`, §4 (spatial routing vs
content processing, associative key-value memory interpretation).
