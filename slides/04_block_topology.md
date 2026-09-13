# A Transformer Layer Is Two Sub-Layers Writing Deltas onto One Residual Stream

## Takeaway
A Transformer layer is not a chain of transformations but a shared state
vector with two write paths: one that moves information between tokens,
one that transforms it within a token.

## Purpose
Opens the chapter and establishes the frame every later slide hangs off.
Each subsequent component ($W^O$, norm placement, FFN, SwiGLU) is
introduced as an answer to a specific failure of this skeleton, so the
skeleton must be on screen first.

## Content
- Layer input and output are both $[B, L, d_{\text{model}}]$.
- Two sub-layers, each structured identically: RMSNorm on the branch,
  sub-layer, additive write back to the stream.
- Sub-layer 1 routes information **across** sequence positions.
- Sub-layer 2 processes information **within** a sequence position.
- Name the routing/processing axis explicitly here; it is the chapter's
  organizing distinction and is developed in `09_routing_vs_processing`.
- The attention branch also contains the output projection, deferred to
  `05_output_projection`.

## Visual
Full-width vector redraw of the source block figure.

- Thick vertical residual spine running top to bottom, unbroken.
- Two branches peel off the spine and re-enter at a `+` node.
- Branch boxes in order: RMSNorm → Attention → $W^O$, then
  RMSNorm → SwiGLU FFN.
- Branch annotations (slide text): "across $L$" on the attention branch,
  "across $d$" on the FFN branch.
- Attention box label (slide text): `Attention (MHA -> GQA, §IV)`.
- Build this as a reusable component: `09` and `12` re-highlight parts of
  the same figure.

## Equations
None. Shape annotations only: $[B, L, d_{\text{model}}]$ at layer input
and output.

## Technical constraints
- The residual spine must be visually continuous through both `+` nodes.
  Nothing may sit on the spine itself.
- The norm boxes sit on the branches only. A norm drawn on the spine
  states the Post-LN architecture, which is wrong for this baseline.
- The branch split is a fan-out of the same tensor to two consumers, not
  a routing or control decision. Do not draw it as a switch or gate.
- The second branch reads the post-attention state. It must not be drawn
  reading the layer input directly.
- No KV cache, no causal mask, no per-head decomposition on this slide.

## Source
`docs/02_Building_The_Complete_Transformer.md`, chapter preamble and the
block diagram (lines 3–37).
