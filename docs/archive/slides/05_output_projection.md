# Concatenating Heads Leaves Them Isolated; $W^O$ Is Where Subspaces Mix

## Takeaway
$W^O$ is not a shape fix — concatenation already yields
$d_{\text{model}}$ — it is the only linear operator that lets one head's
output influence another head's coordinates.

## Purpose
Closes the gap left by the previous lecture: attention produces $h$
independent per-head outputs, and something must turn them into a single
vector that can be written to the residual stream.

## Content
- $d_{\text{model}}$ is partitioned into $h$ heads of width
  $d_k = d_{\text{model}} / h$.
- Concatenating $h$ head outputs already has width $d_{\text{model}}$,
  so the projection is not there to fix dimensions.
- Without $W^O$: head $i$'s output can only reach residual coordinates
  $[i \cdot d_k : (i{+}1) \cdot d_k]$ — head isolation.
- Consequence: no linear recombination of features across heads (the
  source's example: a syntactic head combined with a long-range semantic
  head).
- Concrete shape walk, Llama 3 8B: $d_{\text{model}} = 4096$, $h = 32$,
  $d_k = 128$, $B = 2$, $L = 512$.

## Visual
Split layout. Left: equations. Right: the shape/mixing figure.

- $h$ colored slices → concatenated bar → dense
  $d_{\text{model}} \times d_{\text{model}}$ matrix.
- Small inset contrasting the two reachability patterns: block-diagonal
  (no $W^O$, each head confined to its own coordinate block) against
  dense (with $W^O$). This inset carries the slide's argument and should
  be the visually dominant element on the right.
- Shape chain as annotations under the figure:
  $[2, 512, 128] \to [2, 512, 4096] \to [2, 512, 4096]$.

## Equations
- $\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \dots,
  \text{head}_h)\, W^O$
- $W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$
- Optionally $\text{head}_i = \text{Softmax}\!\left(\frac{Q_i K_i^T}
  {\sqrt{d_k}} + M\right) V_i$ — for notation continuity only, stated as
  given and never derived.

## Technical constraints
- Do not describe $W^O$ as fixing or restoring the dimension. That
  framing is the misconception this slide exists to remove.
- The block-diagonal inset illustrates which residual coordinates each
  head can reach in the absence of $W^O$. It must not suggest that real
  architectures contain a block-diagonal $W^O$ matrix.
- Do not re-derive scaled dot-product attention, the $\sqrt{d_k}$ scale,
  causal masking, or the motivation for multiple heads.
- Do not discuss GQA here; $K$/$V$ head sharing is §IV.

## Source
`docs/02_Building_The_Complete_Transformer.md`, §1 (rationale,
mathematical formulation, concrete tensor example).
