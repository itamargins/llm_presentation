
## 2. LIST OF OMITTED SUBJECTS (Saved for Potential Add-ons)

Here is the tracking list of topics **not** included in this prioritized core session:

* **Computational Efficiency Tricks:**
  * *FlashAttention* (Online Softmax tiling algorithm).
  * *PagedAttention / vLLM* (Virtual memory management for KV cache fragmentation).
  * *Speculative Decoding* (Draft model + target model validation pass).
* **Non-Transformer & Sparse Architectures:**
  * *Sparse Mixture-of-Experts (MoE)* (Top-$k$ routing, shared experts in Mixtral/DeepSeek).
  * *Linear State-Space Models* (Mamba / continuous-to-discrete state space equations).
* **Advanced Memory & Context Extensions:**
  * *Multi-Head Latent Attention (MLA)* (DeepSeek V2/V3 low-rank joint compression $c_t^{KV}$).
  * *RoPE Base Frequency Scaling / YaRN* (Context extension mechanics beyond training length).
* **Cross-Modal Teaser:**
  * *VLM Visual Token Projection* (Mapping vision encoder grids into the causal text decoder stream).

---
