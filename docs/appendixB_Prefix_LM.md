# Prefix-LM vs. Decoder-Only: Why Prefix-LM Lost

<code> I tried asking why decoder-only architecture can't use bi-directional attention for input-prompt prefill phase and then causal-masking for the generated output.   
Turns out this was tried and was less preferred than decoder-only.  
TL;DR - 1. Decoder-only uses the data more efficiently in training, 2. There is Prefix Caching (sharing prompt KV caches across different user requests that start with the same system prompt)
</code>

### Here is the answer from Gemini when discussing this subject:

The exact architecture you are proposing is called Prefix-LM (used in models like PaLM's 20B variant, T5's encoder-decoder reformulation, or UniLM).                      

PREFIX-LM ATTENTION MASK
                      
               Prompt Tokens          Generated Tokens
              ["The", "cat", "sat"]      ["on", "the"]
  
  ["The"]      [  1     1     1  ]     [  0     0  ]  <-- Bi-directional
  ["cat"]      [  1     1     1  ]     [  0     0  ]  <-- Bi-directional
  ["sat"]      [  1     1     1  ]     [  0     0  ]  <-- Bi-directional
  ──────────────────────────────────────────────────
  ["on"]       [  1     1     1  ]     [  1     0  ]  <-- Causal
  ["the"]      [  1     1     1  ]     [  1     1  ]  <-- Causal



While Prefix-LM gives slightly better representations for the prompt, the industry abandoned it for pure Decoder-Only causal models due to two practical realities:

A. Pre-training Data Efficiency (The "Any-Point Cut" Advantage)In pure Causal Decoder training, every single token in a 100-billion-token text corpus acts as a training target. If you feed a 4,096-token document into Llama, the model computes 4,095 next-token prediction losses in a single parallel GPU forward pass.In Prefix-LM training, prompt tokens are masked out from the loss calculation (since they attend bi-directionally, predicting them would be trivial cheating). You only compute loss on the generation suffix.Result: Causal models get $2\times$ to $3\times$ more training signal per byte of text, making them vastly more sample-efficient during expensive multi-million-dollar pre-training runs.

B. Continuous Batching & KV Cache Sharing in ProductionModern LLM serving engines (vLLM, TensorRT-LLM) rely on Prefix Caching (sharing prompt KV caches across different user requests that start with the same system prompt).With Causal Prefill, the KV cache for the system prompt "You are a helpful assistant..." is identical regardless of what user query follows it, because early tokens never look forward to later tokens.With Bi-directional Prefill, if two users send slightly different prompt extensions after the system prompt, the bi-directional attention causes the system prompt's KV vectors to change! This makes cross-request KV cache sharing and multi-turn conversation caching significantly harder to implement efficiently.