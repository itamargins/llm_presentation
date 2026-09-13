# One Block in Six Equations: The Complete Llama-3-Style Forward Pass

## Takeaway
Every component in this chapter composes into six ordered operations, and
Chapter III changes exactly one thing about them — the sequence length
the block is evaluated at.

## Purpose
Chapter recap and reference slide. Also the hand-off: with the full
forward pass on screen, prefill versus decode can be introduced as a
question about how many positions Steps 1–6 are run over.

## Content
- The six steps in order, with their intermediate state names:
  attention pre-norm, attention and output projection, attention
  residual write, FFN pre-norm, SwiGLU FFN, FFN residual write.
- The intermediate state $x^{(l-1/2)}$ names the post-attention,
  pre-FFN state and makes the two-write structure explicit.
- Code mirror of the same six steps as a `TransformerBlock.forward`.
- One closing pointer to Chapter III (no content, just the hand-off).

## Visual
Split layout, math left / code right.

- Left: the six equations, numbered, in execution order.
- Right: `<div class="code-col">` with `TransformerBlock.forward`, max
  10 lines, max 45 characters per line.
- Each code line is color-keyed to its corresponding equation on the
  left. The color correspondence is the slide's visual mechanism and
  must be exact.

## Equations
The six steps as given in the source summary:

1. $\tilde{x}^{(l-1)} = \text{RMSNorm}(x^{(l-1)})$
2. $a^{(l)} = \text{GQA}(\tilde{x}^{(l-1)}) W^O$
3. $x^{(l-1/2)} = x^{(l-1)} + a^{(l)}$
4. $\tilde{x}^{(l-1/2)} = \text{RMSNorm}(x^{(l-1/2)})$
5. $f^{(l)} = \left( \text{SiLU}(\tilde{x}^{(l-1/2)} W_1) \otimes
   (\tilde{x}^{(l-1/2)} W_2) \right) W_3$
6. $x^{(l)} = x^{(l-1/2)} + f^{(l)}$

## Technical constraints
- Intermediate state names must match those used on `06`; this slide is
  the payoff for that notation, and a rename breaks it.
- Do not redraw the block figure from `04`. The equations and code are
  the visual here.
- $\text{GQA}$ appears as an opaque call. Do not expand it, and do not
  explain head grouping — that is §IV.
- No KV cache, no prefill/decode distinction beyond the one-line
  pointer. The slide states the forward pass; Chapter III states what
  happens when it is run per generated token.
- The code must reproduce the same order and the same two residual
  writes. It is a mirror of the equations, not an idiomatic rewrite.

## Source
`docs/02_Building_The_Complete_Transformer.md`, "Comprehensive
Mathematical Summary of a Full Block (Llama 3 Style)".
