# Norm Placement Removes Warm-Up: Pre-LN Leaves the Identity Path Exact

## Takeaway
The training-stability win of modern blocks comes from moving the
normalization onto the branch, not from changing which normalizer is
used.

## Purpose
Separates two things that are usually conflated — where the norm sits
(this slide) and what the norm computes (`08`). The placement argument is
the one that explains zero-warmup training, and it is lost if the two are
merged.

## Content
- Why a norm is needed at all: repeated additive writes make activation
  variance grow monotonically with depth, pushing Softmax inputs into
  saturated, zero-gradient regimes.
- Post-LN (original Transformer): the norm sits on the main residual
  path. Gradients through it are scaled inversely by the norm of the
  activations; activations are large near the output, so early-layer
  updates become tiny. This is what required warm-up schedules.
- Pre-LN (modern standard): the norm sits exclusively on the branch
  entering the sub-layer. The main path stays a pure identity, which is
  what enables stable zero-warmup training at 100B+ scale.

## Visual
Full-width, two minimal graphs side by side, same sub-layer box in both.

- The residual path is stroked in the accent color in both graphs.
- The single visual message: in Post-LN the accent stroke passes
  **through** the norm box; in Pre-LN it passes **around** it.
- Labels (slide text): `Post-LN`, `Pre-LN`, and on the Pre-LN path
  `identity`.
- Nothing else may differ between the two graphs — same box positions,
  same arrow style — so that placement is the only visible variable.

## Equations
- Post-LN: $x^{(l)} = \text{LN}\!\left(x^{(l-1)} + f(x^{(l-1)})\right)$
- Pre-LN: $x^{(l)} = x^{(l-1)} + f\!\left(\text{LN}(x^{(l-1)})\right)$
- Optional, only if it fits without crowding:
  $\text{Var}(x^{(l)}) \approx \text{Var}(x^{(0)}) +
  \sum \text{Var}(\Delta x)$

## Technical constraints
- This slide is about placement. Do not introduce RMSNorm here; use a
  generic `LN` operator so the contrast stays clean.
- Keep the variance relation as an approximation. It assumes independence
  of the per-layer deltas and must not be written as an equality.
- Do not explain what a learning-rate warm-up schedule is, and do not
  invoke internal covariate shift.
- The Post-LN graph must still show a residual addition. The difference
  is where the norm is applied, not whether a residual exists.

## Source
`docs/02_Building_The_Complete_Transformer.md`, §3 (rationale,
Post-LN vs Pre-LN evolution).
