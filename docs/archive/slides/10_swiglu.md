# SwiGLU Buys a Multiplicative Gate and Pays for It by Shrinking $d_{\text{ff}}$ to $\tfrac{8}{3} d_{\text{model}}$

## Takeaway
The third weight matrix is budget-neutral only because $d_{\text{ff}}$ is
rescaled, so gating is a parameter-matched swap rather than added
capacity.

## Purpose
Explains the one part of the modern FFN that differs from the classical
two-matrix MLP, and does so as an accounting argument rather than an
activation-function preference.

## Content
- GLU structure: two parallel linear projections of the same input, one
  passed through SiLU, combined by an elementwise product, then
  projected down.
- Three matrices: $W_1$ gate projection, $W_2$ up projection, $W_3$ down
  projection.
- SiLU (Swish-1) is the gate non-linearity.
- Parameter accounting: a two-matrix FFN at $d_{\text{ff}} =
  4 d_{\text{model}}$ costs $8 d_{\text{model}}^2$; three matrices at
  $d_{\text{ff}} = \tfrac{8}{3} d_{\text{model}}$ cost the same.
- $d_{\text{ff}}$ is rounded to a hardware-aligned multiple (the source
  cites multiples of 256).
- Adopted by PaLM and Llama 3.

## Visual
Split layout. Left: equations and the parameter arithmetic. Right: the
three-branch dataflow.

- Input splits into two parallel `Linear` boxes ($W_1$, $W_2$).
- SiLU sits on the $W_1$ branch only.
- Both branches meet at an $\otimes$ node, then a single `Linear` ($W_3$).
- Width annotation (slide text): `d_ff` on the edges between the
  $\otimes$ node and $W_3$, `d_model` on input and output.

## Equations
- $\text{SwiGLU}(x) = \left( \text{SiLU}(x W_1) \otimes x W_2 \right) W_3$
- $\text{SiLU}(x) = x \cdot \sigma(x)$
- $3 \times \left( \tfrac{8}{3} d_{\text{model}} \times d_{\text{model}}
  \right) = 8 d_{\text{model}}^2$

## Technical constraints
- **Gate convention.** Follow `docs/02`: SiLU is applied to the $W_1$
  branch. `docs/00_high_level_plan.md` line 53 gates the opposite branch;
  the two are equivalent up to relabeling but the deck must use one
  convention. Do not mix them.
- **Exclude the Llama 3 8B $d_{\text{ff}}$ claim.** The source states
  $14336 \approx \tfrac{8}{3} \times 4096$, but
  $\tfrac{8}{3} \times 4096 = 10923$ and $14336 = 3.5 \times 4096$.
  Llama 3 8B is therefore wider than parameter-matched and contradicts
  the rule this slide states. Present the $\tfrac{8}{3}$ rule
  abstractly. A concrete model instance can only be added once a source
  for the additional width multiplier exists.
- The $\otimes$ node is an elementwise join of two tensors of equal
  shape. It must not be drawn as a switch, router, or control decision —
  "gate" here is multiplicative, not conditional.
- The two branches are parallel projections of the same input; do not
  draw them as sequential.
- Do not show activation-function curves or compare ReLU/GELU/SiLU
  shapes.

## Source
`docs/02_Building_The_Complete_Transformer.md`, §4 (evolution to SwiGLU,
parameter-matching strategy).
