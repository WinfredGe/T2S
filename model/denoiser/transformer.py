import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import numpy as np
import math

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################

def get_sinusoidal_positional_embeddings(num_positions, d_model):
    position = torch.arange(num_positions).unsqueeze(1)  # shape: (num_positions, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).unsqueeze(
        0)  # shape: (1, d_model/2)

    # 计算编码
    pos_embedding = torch.zeros(num_positions, d_model)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)  # 奇数维度
    pos_embedding[:, 1::2] = torch.cos(position * div_term)  # 偶数维度

    return pos_embedding.unsqueeze(0)  # 为 batch size 添加一维

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        assert dim % 2 == 0, "Dimension must be even"
    def forward(self, t):
        # 扩展时间尺度
        t = t * 100.0
        t = t.unsqueeze(-1)

        # 生成频率
        freqs = torch.pow(10000, torch.linspace(0, 1, self.dim // 2)).to(t.device)

        # 计算sin和cos嵌入
        sin_emb = torch.sin(t[:, None] / freqs)
        cos_emb = torch.cos(t[:, None] / freqs)
        # 合并sin和cos嵌入
        embedding = torch.cat([sin_emb, cos_emb], dim=-1)
        embedding = embedding.squeeze(1)

        # 通过MLP进行处理
        return embedding

################################################
#               Embedding Layers               #
################################################

class LatentEmbedding(nn.Module):
    def __init__(self, embed_dim: int=64):
        super().__init__()
        self.dim = embed_dim
        self.embedding2d = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=(6, 6),  # 卷积窗口为 6x6
            stride=(6, 6),  # 每隔 6 个单位进行一次卷积
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        B, _, M, N = x.shape
        x = self.embedding2d(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class InverseLatentEmbedding(nn.Module):
    def __init__(self, embed_dim: int=64):
        super().__init__()
        self.dim = embed_dim
        self.inv_embedding2d = nn.ConvTranspose2d(
            in_channels=embed_dim,    # 输入通道数为 embed_dim
            out_channels=1,           # 输出通道数为 1
            kernel_size=(6, 6),       # 卷积窗口为 6x6
            stride=(6, 6),            # 每隔 6 个单位进行一次卷积
        )
        self.fc1 = nn.Linear(60, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        B, K, C = x.shape
        x = x.transpose(1, 2).reshape(B, self.dim, 1, K)
        x = self.inv_embedding2d(x)
        x = x.squeeze(1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1)
        return x


#################################################################################
#                                 Core Model                                #
#################################################################################

class Transformerlayer(nn.Module):
    def __init__(self, ):
        super().__init__()
        d_model = 128 #64
        mlp_ratio = 2.0
        mlp_hidden_dim = int(d_model * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d_model, num_heads=4, qkv_bias=True)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )
        # self.fc1 = nn.Linear(64, 128)
        # self.fc2 = nn.Linear(128, 64)


    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))

        # x = self.fc1(x)
        # x = torch.relu(x)
        # x = self.fc2(x)


        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        # patchify
        self.channel=1
        self.H = 30 #6 # 30
        self.W = 64
        emb_size=128 #64
        self.patch_size=2
        self.patch_count=int((self.H/self.patch_size)*(self.W/self.patch_size))
        self.conv=nn.Conv2d(in_channels=self.channel,out_channels=self.channel*self.patch_size**2,kernel_size=self.patch_size,padding=0,stride=self.patch_size)
        self.patch_emb=nn.Linear(in_features=self.channel*self.patch_size**2,out_features=emb_size)
        pos_embed = get_sinusoidal_positional_embeddings(self.patch_count, emb_size)
        self.pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)
        self.ln = nn.LayerNorm(emb_size)
        self.linear_emb_to_patch = nn.Linear(emb_size, self.channel * self.patch_size ** 2)


        self.time_emb = TimeEmbedding(dim=emb_size)
        # pos_embed = get_sinusoidal_positional_embeddings(6,64)
        # self.pos_embed = torch.nn.Parameter(pos_embed, requires_grad=False)

        self.layers = nn.ModuleList([Transformerlayer() for _ in range(4)])
        self.unpatch = InverseLatentEmbedding(embed_dim=emb_size)



        self.initialize_weights()



    def forward(self, input: torch.Tensor, t: torch.Tensor, text_input):
        """
                x: (B, M, N) tensor of input latent (batch, latent num:4, latent dim:64)
                t: (B,) tensor of diffusion timesteps
                text_input:
                """
        # x = input.permute(0, 2, 1)
        # x = x + self.pos_embed
        x = input.permute(0, 2, 1)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (batch,new_channel,patch_count,patch_count)
        x = x.permute(0, 2, 3, 1)  # (batch,patch_count,patch_count,new_channel)
        x = x.view(x.size(0), self.patch_count, x.size(3))  # (batch,patch_count**2,new_channel)
        x = self.patch_emb(x)  # (batch,patch_count**2,emb_size)
        x = x + self.pos_embed  # (batch,patch_count**2,emb_size)

        t = self.time_emb(t)

        c = t
        if text_input is not None:
            c = t + text_input
        for layer in self.layers:
            x = layer(x, c)

        x = self.ln(x)
        x = self.linear_emb_to_patch(x)
        x = x.view(x.size(0), int(self.H/self.patch_size), int(self.W/self.patch_size), self.channel, self.patch_size, self.patch_size)
        x = x.permute(0, 3, 1, 2, 4, 5)  # (batch,channel,patch_count(H),patch_count(W),patch_size(H),patch_size(W))
        x = x.permute(0, 1, 2, 4, 3, 5)  # (batch,channel,patch_count(H),patch_size(H),patch_count(W),patch_size(W))
        x = x.reshape(x.size(0), self.channel, self.H,
                      self.W)  # (batch,channel,img_size,img_size)
        x = x.squeeze(1)
        x = x.permute(0, 2, 1)


        return x
    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)