import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5EncoderModel
from typing import List, Tuple, Optional

class TextEncoder(nn.Module):
    """
    Text encoder using mT5-XXL as specified in the paper section 2.2.1
    """
    def __init__(self, dim: int = 512):
        super().__init__()
        # Load mT5-XXL encoder
        self.mt5 = MT5EncoderModel.from_pretrained('google/mt5-xxl')
        # Project mT5 features to model dimension
        self.proj = nn.Linear(self.mt5.config.hidden_size, dim)
        
    def forward(self, text: str) -> torch.Tensor:
        # Get mT5 features
        mt5_output = self.mt5(text)
        # Project to model dimension
        return self.proj(mt5_output.last_hidden_state)

class TransformerBlock(nn.Module):
    """
    Transformer block as shown in Figure 4(b)
    """
    def __init__(self, dim: int, use_full_attn: bool = True):
        super().__init__()
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # Attention layers
        if use_full_attn:
            self.self_attn = MultiheadAttention(dim)
        else:
            self.self_attn = SiparseAttention(dim)
            
        self.cross_attn = MultiheadAttention(dim)
        
        # Scale and shift parameters from timestep as in paper
        self.scale1 = nn.Linear(dim, dim * 2)
        self.scale2 = nn.Linear(dim, dim * 2)
        self.scale3 = nn.Linear(dim, dim * 2)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, 
                text_emb: torch.Tensor) -> torch.Tensor:
        # Get scale and shift parameters
        scale1, shift1 = self.scale1(timestep).chunk(2, dim=-1)
        scale2, shift2 = self.scale2(timestep).chunk(2, dim=-1)
        scale3, shift3 = self.scale3(timestep).chunk(2, dim=-1)
        
        # Self attention
        h = self.norm1(x)
        h = h * scale1 + shift1
        h = self.self_attn(h) + x
        
        # Cross attention
        h2 = self.norm2(h)
        h2 = h2 * scale2 + shift2
        h2 = self.cross_attn(h2, text_emb) + h
        
        # Feed forward
        h3 = self.norm3(h2)
        h3 = h3 * scale3 + shift3
        h3 = self.ffn(h3) + h2
        
        return h3

class MultiheadAttention(nn.Module):
    """
    Standard multihead attention for full attention layers
    """
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # Get query, key, value
        if context is None:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
        else:
            q = self.qkv(x).reshape(B, N, self.num_heads, self.head_dim)
            kv = self.qkv(context).reshape(B, context.shape[1], 2, 
                                         self.num_heads, self.head_dim)
            k, v = kv.unbind(2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block for structure controller as in Figure 8
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, dim)
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        h = self.act(h)
        return h + x  # residual connection

class TokenTransform(nn.Module):
    """
    Token-wise transformation for structure controller
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim * 4, dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.linear1(h)
        h = self.act(h)
        h = self.linear2(h)
        return h + x  # residual connection

# Additional required functions
def apply_min_max_tokens(batch: dict, max_tokens: int = 65536) -> dict:
    """
    Implements min-max token strategy from Section 3.1
    """
    # Get resolution stride
    s = 16  # as specified in paper
    
    # Calculate tokens for each sample
    token_counts = []
    for x in batch['video']:
        h, w = x.shape[-2:]
        tokens = (h * w) // (s * s)
        token_counts.append(tokens)
        
    # Find samples that fit within max tokens
    valid_indices = [i for i, t in enumerate(token_counts) 
                    if t <= max_tokens]
    
    # Filter batch
    filtered_batch = {
        k: [v[i] for i in valid_indices]
        for k, v in batch.items()
    }
    
    return filtered_batch

def clip_gradients_adaptive(model: nn.Module, 
                          loss: torch.Tensor,
                          alpha: float = 0.99) -> None:
    """
    Implements adaptive gradient clipping from Section 3.2
    """
    # Calculate gradient norm
    grad_norm = torch.norm(
        torch.stack([
            p.grad.norm(p=2)
            for p in model.parameters()
            if p.grad is not None
        ])
    )
    
    # Update EMA
    if not hasattr(model, 'grad_ema'):
        model.grad_ema = grad_norm
    model.grad_ema = alpha * model.grad_ema + (1 - alpha) * grad_norm
    
    # Clip if gradient norm is too large
    clip_factor = (model.grad_ema / grad_norm).clamp(max=1.0)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.mul_(clip_factor)