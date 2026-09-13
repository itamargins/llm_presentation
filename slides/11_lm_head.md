# The LM Head Closes the Loop: Final Norm, Unembedding, Temperature-Scaled Softmax

## Takeaway
The entire block stack produces one vector per position, and three
operations turn the last one into the distribution that drives
autoregressive generation.

## Purpose
Resolves the block-to-token-prediction link and hands off to Chapter III:
once generation is a per-step loop over this head, the decode bottleneck
becomes the obvious next question.

## Content
- The final hidden state $x^{(L)} \in \mathbb{R}^{B \times L \times
  d_{\text{model}}}$ is the fully contextualized sequence representation.
- A final RMSNorm is applied before the head; it is distinct from the
  per-block norms.
- Unembedding projection to vocabulary size:
  $W_{\text{vocab}} \in \mathbb{R}^{d_{\text{model}} \times |V|}$,
  producing logits $z_t \in \mathbb{R}^{|V|}$.
- Weight tying: some architectures share $W_{\text{vocab}}$ with
  $W_{\text{embed}}^T$; modern large models such as Llama 3 often untie
  them so the output projection can optimize independently of the input
  representation.
- Temperature $T > 0$ scales the logits before Softmax.
- Three regimes: $T \to 0$ collapses onto the argmax; $T = 1$ preserves
  the learned distribution; $T > 1$ flattens toward uniform, increasing
  sample variance.
- At generation time only the vector at the last position is projected.

## Visual
Split layout.

- Left: vertical pipeline, $x_t^{(L)} \to$ RMSNorm $\to W_{\text{vocab}}
  \to z_t \to /T \to$ Softmax $\to P(x_{t+1})$, with widths annotated
  ($d_{\text{model}}$, then $|V|$).
- Right: a small three-panel logit/probability bar plot at $T \to 0$,
  $T = 1$, $T > 1$ over the same handful of candidate tokens. This is the
  element that makes temperature land; the pipeline alone is text
  arranged vertically.

## Equations
- $z_t = \text{RMSNorm}\!\left(x_t^{(L)}\right) W_{\text{vocab}}
  \in \mathbb{R}^{|V|}$
- $P(x_{t+1} = i \mid x_{1:t}) = \dfrac{\exp(z_{t,i}/T)}
  {\sum_{j=1}^{|V|} \exp(z_{t,j}/T)}$

## Technical constraints
- Temperature divides the **logits before** Softmax, never the
  probabilities after. The pipeline order must show this.
- $T \to 0$ is a limit. Do not present a literal division by zero;
  greedy decoding is implemented as an argmax.
- The bar plots must use the same underlying logits across all three
  panels, otherwise they illustrate nothing.
- Do not introduce top-$k$, top-$p$, beam search, or repetition
  penalties — none are in the source document.
- Do not discuss the training objective or cross-entropy loss.
- The source's "hallucination risk" wording at $T > 1$ is a qualitative
  aside; keep the slide's claim to entropy and sample variance.
- Do not revisit tokenization or input embeddings (Chapter I).

## Source
`docs/02_Building_The_Complete_Transformer.md`, §5 (unembedding
projection, Softmax and temperature scaling, and the head pipeline
diagram).
