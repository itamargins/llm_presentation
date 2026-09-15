# Lecture Timing Breakdown

## Section Reference Table

| # | Section | Estimated Time | Cumulative | Goal | Subtopics (time) | Key Teaching Points |
|---|---|---:|---:|---|---|---|
| 1 | Opening and Roadmap | 4 min | 4 min | Set context and map the lecture flow to the 7-part deck. | Cover + agenda framing (2 min)<br>Bridge from prior MHA lecture (2 min) | Frame the problem: moving from isolated MHA to a scalable 80+ layer inference pipeline. |
| 2 | Attention -> Transformer Block | 24 min | 28 min | Build a stable deep block from MHA outputs. | Output projection and tensor shapes (5 min)<br>Residual paths and gradient highway (5 min)<br>Normalization: LayerNorm/RMSNorm, Post-LN vs Pre-LN (8 min)<br>FFN, SwiGLU gating, and parameter matching (6 min) | Show [B,L,h*dk] -> W_O -> [B,L,d_model]. Explain residual +I gradient path. Compare Pre-LN stability vs Post-LN scaling issues. Show why d_ff = (8/3)d_model keeps SwiGLU parameter count matched. |
| 3 | Encoder-Decoder Architecture and Variants | 10 min | 38 min | Contrast encoder-only, encoder-decoder, and decoder-only designs. | Encoder-only vs enc-dec vs decoder-only (7 min)<br>Causal mask equation and implementation (3 min) | Connect objective and masking to architecture choice. Use the upper-triangular mask with -inf to enforce past-only attention in training and prefill. |
| 4 | Language Modeling and Generation | 11 min | 49 min | Map hidden states to tokens and explain decoding controls. | LM head and unembedding (3 min)<br>Temperature and logit scaling (4 min)<br>Nucleus (top-p) sampling flow (4 min) | Cover T -> 0 (greedy), T = 1, T > 1. Explain top-p flow: sort -> cumulative cutoff -> renormalize -> categorical sample. |
| 5 | LLM Execution Phases | 8 min | 57 min | Explain why prefill and decode behave differently on GPU hardware. | Prefill compute profile (GEMM, 4 min)<br>Decode memory profile (GEMV, VRAM/SRAM traffic, 4 min) | Prefill uses parallel matrix-matrix math. Decode is serial and bandwidth-limited by repeated weight movement from VRAM to SRAM. |
| 6 | KV Cache | 12 min | 69 min | Motivate KV cache and quantify its memory trade-off. | Why KV cache is needed (4 min)<br>KV cache formula and footprint (4 min)<br>Llama-3 70B reference + demo pointer (4 min) | Show naive O(L^2) recompute cost, then bytes formula: 2*b*s*l*h_kv*d_h*P and its VRAM implications. |
| 7 | The Context Window | 8 min | 77 min | Explain long-context scaling limits and why they matter operationally. | Context-window definition and boundary (3 min)<br>Compute O(L^2), KV memory O(L), positional/OOD effects (5 min) | Tie context length directly to latency growth, memory pressure, and positional generalization limits. |
| 8 | Memory Optimization Techniques | 10 min | 87 min | Show methods that reduce KV or attention memory/runtime cost. | MQA/GQA mechanics and trade-off (4 min)<br>GQA conversion note + adoption (2 min)<br>MLA, PagedAttention, FlashAttention overview (4 min) | MQA/GQA reduce KV heads. MLA compresses KV state. PagedAttention reduces fragmentation. FlashAttention reduces HBM traffic via tiled exact attention. |
| 9 | Wrap-up and Synthesis | 3 min | 90 min | Close with end-to-end system perspective and next-step pointer. | Summary narrative (3 min) | Reinforce pipeline-level thinking: block design -> decoding mechanics -> systems bottlenecks -> optimization toolkit. |
| **Total** | **Full Lecture** | **~90 min** | **1.5 hours** | - | - | - |

## Delivery and Time-Cut Guide

| Scenario | Adjustment |
|---|---|
| Need to fit a 75-minute slot | Compress Section 2 to a 16-minute version (keep one normalization comparison and one SwiGLU slide).<br>Fold Section 7 into a 4-minute summary inside Section 6.<br>Keep Section 8 focused on MQA/GQA + PagedAttention; move MLA, FlashAttention, and hfviewer to optional/appendix time. |
| Interactive check-in (around minute 57) | Ask: "If prompt prefill processes 1,000 tokens in 50 ms, why can generating 100 tokens still take around 2,000 ms?" |
