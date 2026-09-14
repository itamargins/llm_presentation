# Lecture Timing Breakdown

## Section Reference Table

| # | Section | Estimated Time | Cumulative | Goal | Subtopics (time) | Key Teaching Points |
|---|---|---:|---:|---|---|---|
| 1 | Introduction and MHA Recap | 5 min | 5 min | Re-anchor from Lecture 1 and transition from single MHA to stacked blocks. | Core transformer diagram walkthrough (5 min). | MHA leaves independent head outputs; we need a stable pipeline for 80+ layers. |
| 2 | Attention -> Transformer Block | 25 min | 30 min | Build the full block from MHA output to deep-stack stability. | Output projection and tensor shapes (5 min)<br>Residuals and gradient highway (6 min)<br>Normalization: Pre-LN vs Post-LN and RMSNorm (8 min)<br>FFN and SwiGLU mechanics (6 min) | Show [B,L,h*dk] -> W_O -> [B,L,d_model]. Explain residual +I gradient path. Compare Pre-LN stability vs Post-LN scaling issues. Show why d_ff = (8/3)d_model keeps SwiGLU parameter count matched. |
| 3 | Encoder-Decoder Architecture Variants | 10 min | 40 min | Contrast encoder-only, encoder-decoder, and decoder-only paths. | BERT vs Enc-Dec vs GPT/Llama (5 min)<br>Causal masking implementation (5 min) | Use the upper-triangular mask with -inf to enforce past-only attention during training and prefill. |
| 4 | Language Modeling and Generation Mechanics | 12 min | 52 min | Map hidden states to tokens and explain decoding controls. | LM head and unembedding (3 min)<br>Temperature and logit scaling (4 min)<br>Nucleus (top-p) sampling (5 min) | Cover T -> 0 (greedy), T = 1, T > 1. Explain top-p flow: sort -> cumulative cutoff -> renormalize -> categorical sample. |
| 5 | LLM Execution Phases and Hardware Realities | 10 min | 62 min | Explain why prefill and decode behave differently on GPU hardware. | Prefill (compute-bound GEMM, 4 min)<br>Decode (memory-bound GEMV, 6 min) | Prefill uses parallel matrix-matrix math. Decode is serial and bandwidth-limited by VRAM -> SRAM weight movement each token. |
| 6 | The Key-Value (KV) Cache | 13 min | 75 min | Motivate KV cache and quantify its memory trade-off. | Why KV cache is needed (4 min)<br>KV cache formula and footprint (5 min)<br>Live demo: time vs memory (4 min) | Show naive O(L^2) recompute cost, then bytes formula: 2*b*s*l*h_kv*d_h*P. Use Llama-3 70B example to show large VRAM cost. |
| 7 | Bypassing the Memory Wall (Advanced Optimizations) | 12 min | 87 min | Show architecture and kernel methods that reduce KV or attention cost. | MQA and GQA (4 min)<br>DeepSeek MLA (3 min)<br>PagedAttention (3 min)<br>FlashAttention tiling (2 min) | MQA/GQA reduce KV heads. MLA compresses KV into low-rank latent state. PagedAttention avoids fragmentation. FlashAttention tiles SRAM and avoids materializing full NxN attention. |
| 8 | Wrap-up and Synthesis | 3 min | 90 min | Close with end-to-end system perspective. | Summary narrative (3 min). | Tie all components into a production-ready inference pipeline. |
| **Total** | **Full Lecture** | **~90 min** | **1.5 hours** | - | - | - |

## Delivery and Time-Cut Guide

| Scenario | Adjustment |
|---|---|
| Need to fit a 75-minute slot | Reduce Section 3 to a 3-minute high-level summary.<br>Move MLA and FlashAttention to appendix/optional slides and keep focus on GQA and PagedAttention. |
| Interactive check-in (around minute 55) | Ask: "If prompt prefill processes 1,000 tokens in 50 ms, why can generating 100 tokens still take around 2,000 ms?" |
