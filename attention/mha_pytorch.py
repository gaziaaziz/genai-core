import torch
from torch import nn
import numpy as np


class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, X):
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        batch_size, seq_len, embed_dim = Q.shape
        head_dim = embed_dim // self.num_heads

        Q = Q.view(batch_size, seq_len, self.num_heads, head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, head_dim)

        Q = Q.permute(0,2,1,3)
        K = K.permute(0,2,1,3)
        V = V.permute(0,2,1,3)

        attn_scores = Q @ K.transpose(-2,-1) / np.sqrt(head_dim)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        mha = attn_scores @ V

        mha = mha.permute(0,2,1,3)

        mha = mha.reshape(batch_size, seq_len, embed_dim)

        output = self.out(mha)


        return output
