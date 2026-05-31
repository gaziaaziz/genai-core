import torch
from torch import nn
import math


class MultiHeadAttn(nn.Module):
    def __init__(self, embed_dim= 512, num_heads= 16):
        super().__init__()
        if not(embed_dim % self.num_heads == 0):
            raise ValueError("Model dimensions must be divisible by the number of heads.")
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // self.num_heads
        self.q_proj = nn.Linear(in_features= embed_dim, out_features= embed_dim, bias= False)
        self.k_proj = nn.Linear(in_features= embed_dim, out_features= embed_dim, bias= False)
        self.v_proj = nn.Linear(in_features= embed_dim, out_features= embed_dim, bias= False)

        self.out_proj = nn.Linear(in_features= embed_dim, out_features= embed_dim, bias= False)

    def forward(self, X):
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)
        batch_size, seq_len, embed_dim = X.shape

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Q = (batch_size, seq_len, num_heads, head_dim)

        Q = Q.permute(0,2,1,3)
        K = K.permute(0,2,1,3)
        V = V.permute(0,2,1,3)
        # Q = (batch_size, num_heads, seq_len, head_dim)
        # K = (batch_size, num_heads, seq_len, head_dim)
        # K.transpose(-2,-1) = (batch_size, num_heads, head_dim, seq_len)
        # QK.T = (batch_size, num_heads, seq_len, seq_len)



        attn_scores = (Q @ K.transpose(-2,-1)) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = attn_weights @ V

        output = output.permute(0,2,1,3)

        output = output.reshape(batch_size, seq_len, embed_dim)

        return self.out_proj(output)
