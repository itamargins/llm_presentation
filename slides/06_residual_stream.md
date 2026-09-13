# Sub-Layers Read the Residual Stream and Write Deltas; the Identity Path Is Never Touched

## Takeaway
The hidden state is an accumulator rather than a pipeline value, which is
simultaneously the gradient argument for extreme depth and the mental
model for every component that follows.

## Purpose
Converts the residual connection from "a gradient trick" into the
shared-bus abstraction the rest of the chapter depends on. Norm
placement (`07`) and the FFN's per-token write (`09`) are only
meaningful once the stream is understood as shared state.

## Content
- Read / compute $\Delta x$ / write semantics: a sub-layer reads the
  current state, computes a delta, and adds it back.
- The stream is only ever added to, never overwritten.
- Motivation for the framing: at 80+ layers, a purely sequential
  $x^{(l)} = f(x^{(l-1)})$ forces every layer to re-learn the identity
  when no transformation is needed.
- Unrolling the recursion turns the layer stack into a sum of deltas on
  the initial state.
- The identity term in the Jacobian gives a gradient path whose
  magnitude does not decay with depth.

## Visual
Split layout. Left: bus diagram. Right: the two equations.

- Horizontal residual bus spanning the full width, drawn continuous.
- Two taps hanging below it, one per sub-layer.
- Arrow typing is the point of the diagram: downward arrow into a
  sub-layer = read (tensor flow); upward arrow back to the bus = additive
  write, terminating in a `+` node on the bus.
- Arrow labels (slide text): `read`, `Δx`, `+`.

## Equations
- $x^{(L)} = x^{(0)} + \sum_{l=1}^{L} \Delta x^{(l)}$
- $\dfrac{\partial \mathcal{L}}{\partial x^{(0)}} =
  \dfrac{\partial \mathcal{L}}{\partial x^{(L)}}
  \left( I + \sum_{l=1}^{L}
  \dfrac{\partial \Delta x^{(l)}}{\partial x^{(0)}} \right)$
- If a single-layer form is shown, it must be the two-write sequential
  form: $x^{(l-1/2)} = x^{(l-1)} + a^{(l)}$, then
  $x^{(l)} = x^{(l-1/2)} + f^{(l)}$.

## Technical constraints
- **Do not reproduce the source's line-85 equation**
  $x^{(l)} = x^{(l-1)} + \text{Attention}(x^{(l-1)}) + \text{FFN}(x^{(l-1)})$.
  The FFN reads the post-attention state, not the layer input. The
  source's own Steps 4–6 have this right; line 85 does not.
- Do not describe $\Delta x$ as sparse. The source uses "sparse
  update/delta vector" but establishes no sparsity result.
- The bus must not pass through any box. A sub-layer drawn inline on the
  bus contradicts the entire slide.
- Do not teach the chain rule, vanishing gradients as a phenomenon, or
  ResNet history. Name them and move on.

## Source
`docs/02_Building_The_Complete_Transformer.md`, §2 (rationale and
mechanics, mathematical and gradient impact).
