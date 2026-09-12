# Notes:

## MHA
- Multi-head attention conceptually splits each embedding equally among the heads, but in practice, each head is served the full embedding and projects it down to the smaller space.

    - For example: original d_model = 512, num_heads = 8, so each head attends to a 512/8 = 64 sized embedding. Instead of splitting 512 to 8 equal pieces, each 512 is projected down to 64 using 8 matrices. This is done for Q, K and V, so 3*8 matrics, each shaped [512x64].

    - A parallel implementation uses one large matrix [512x512], and the output is split to num_heads equally.

