import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        assert dim % 2 == 0, "Dimension must be even"
    def forward(self, t):
        t = t * 100.0
        t = t.unsqueeze(-1)
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        embedding = embedding.squeeze(1)
        return embedding


class TextToSeriesCrossAttention(nn.Module):
    def __init__(self,
                 n_embd,  # the embed dim
                 condition_embd,  # condition dim768
                 n_head,  # the number of heads
                 ):
        super().__init__()
        assert n_embd % n_head == 0
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
    def forward(self, x, encoder_output, mask=None):
        B, T, _ = x.size()
        B, T_E, C_E = encoder_output.size()
        C = self.query.out_features
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) 
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  
        att = F.softmax(att, dim=-1)  
        y = att @ v  
        y = y.transpose(1, 2).contiguous().view(B, T, C)  
        att = att.mean(dim=1, keepdim=False)  
        y = self.proj(y)
        return y, att

class MLPlayer(nn.Module):
    def __init__(self,):
        super().__init__()
        dim = 64
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(6, elementwise_affine=True)
        self.time_emb = TimeEmbedding(dim=dim)
        self.pos_emb = nn.Embedding(6*2, embedding_dim=dim,dtype=torch.float32)
        self.self_attn = nn.MultiheadAttention(dim, 4)
        self.self_attn2 = nn.MultiheadAttention(6, 2)
        self.cross_attn = TextToSeriesCrossAttention(dim, 128, n_head=4)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, dim)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(6, 256),
            nn.ReLU(),
            nn.Linear(256, 6)
        )
    def forward(self, input, t, text_input):

        t = self.time_emb(t)
        t = t.unsqueeze(-1)
        x = input + t
        x = x.permute(0, 2, 1)
        if text_input is not None:
            text_emb = text_input.unsqueeze(1).repeat(1, 6, 1)
            cross_attn, _ = self.cross_attn(x, text_emb, text_emb)
            x = x + cross_attn
        x = self.norm2(x)
        x = x + self.mlp(x)
        x = x.permute(0, 2, 1)
        x = self.mlp2(x)
        return x

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MLPlayer() for _ in range(8)])
    def forward(self, input, t, text_input):
        for layer in self.layers:
            input = layer(input, t, text_input)
        return input
