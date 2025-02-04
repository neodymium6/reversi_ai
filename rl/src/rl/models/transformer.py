import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=2, in_channels=2, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input:  (batch_size, 2, 8, 8)
        # Output: (batch_size, (8 // patch_size) ** 2, embed_dim)
        x: torch.Tensor = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        # Input:  (batch_size, (8 // patch_size) ** 2, embed_dim)
        # Output: (batch_size, (8 // patch_size) ** 2, embed_dim)
        batch_size, n_patches, embed_dim = x.shape
        qkv: torch.Tensor = self.qkv(x)
        qkv: torch.Tensor = qkv.reshape(batch_size, n_patches, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(batch_size, n_patches, embed_dim)
        x = self.proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, 
                 patch_size=2,
                 embed_dim=128,
                 num_heads=4,
                 num_layers=8,
                 mlp_ratio=2.0,
                 dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size, 2, embed_dim)
        
        # Position embeddings for 16 patches (8x8 board with 2x2 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, (8 // patch_size) ** 2, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output layers for value and advantage streams (Dueling architecture)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * (8 // patch_size) ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.advantage_head = nn.Sequential(
            nn.Linear(embed_dim * (8 // patch_size) ** 2, 256),
            nn.ReLU(),
            nn.Linear(256, 65)  # 64 positions + 1 pass move
        )
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        # Patch embedding
        x = self.patch_embed(x)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
            
        x: torch.Tensor = self.norm(x)
        
        # Flatten for the heads
        x = x.reshape(x.shape[0], -1)
        
        # Dueling network architecture
        value: torch.Tensor = self.value_head(x)
        value = value.expand(-1, 65)
        advantage: torch.Tensor = self.advantage_head(x)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True).expand(-1, 65))
        
        return q_values
