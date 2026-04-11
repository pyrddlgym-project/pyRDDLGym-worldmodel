import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, cast

Tensor = torch.Tensor


class SinePositionalEncoding(nn.Module):
    '''Implements absolute positional encoding as described in "Attention is All You Need".'''

    def __init__(self, d_model: int, max_len: int=256, base: float=10000.0) -> None:
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)   # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(base) / d_model))  # (d_model / 2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:x.size(1)]   # (batch, seq_len, d_model)


class RotaryPositionalEmbedding(nn.Module):
    '''Applies rotary positional embeddings to per-head query and key tensors.'''

    def __init__(self, head_dim: int, max_len: int=256, base: float=10000.0) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(f"ROPE requires an even head dimension, got {head_dim}.")

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def _rotate_half(self, x: Tensor) -> Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        seq_len = q.size(-2)
        cos_cache = cast(Tensor, self.cos)
        sin_cache = cast(Tensor, self.sin)
        cos = cos_cache[:seq_len].to(device=q.device, dtype=q.dtype)
        sin = sin_cache[:seq_len].to(device=q.device, dtype=q.dtype)
        cos = torch.repeat_interleave(cos, 2, dim=-1).unsqueeze(0).unsqueeze(0)
        sin = torch.repeat_interleave(sin, 2, dim=-1).unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class RotaryMultiheadAttention(nn.Module):
    '''Minimal self-attention module with rotary positional embeddings on q and k.'''

    def __init__(self, embed_dim: int, num_heads: int, dropout: float=0.0,
                 bias: bool=True, batch_first: bool=True, max_len: int=256) -> None:
        super().__init__()

        if not batch_first:
            raise ValueError("RotaryMultiheadAttention only supports batch_first=True.")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}).")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        self._qkv_same_embed_dim = True

        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rotary = RotaryPositionalEmbedding(self.head_dim, max_len=max_len)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                key_padding_mask: Optional[Tensor]=None, need_weights: bool=True,
                attn_mask: Optional[Tensor]=None, average_attn_weights: bool=True,
                is_causal: bool=False) -> Tuple[Tensor, Union[Tensor, None]]:
        if key_padding_mask is not None:
            raise NotImplementedError("RotaryMultiheadAttention expects padding to be "
                                      "encoded in attn_mask.")
        if not self.batch_first:
            raise ValueError("RotaryMultiheadAttention only supports batch_first=True.")

        batch, query_len, _ = query.shape
        key_len = key.size(1)

        q = F.linear(query, self.in_proj_weight[:self.embed_dim],
                     None if self.in_proj_bias is None 
                     else self.in_proj_bias[:self.embed_dim])
        k = F.linear(key, self.in_proj_weight[self.embed_dim:2 * self.embed_dim],
                     None if self.in_proj_bias is None 
                     else self.in_proj_bias[self.embed_dim:2 * self.embed_dim])
        v = F.linear(value, self.in_proj_weight[2 * self.embed_dim:],
                     None if self.in_proj_bias is None 
                     else self.in_proj_bias[2 * self.embed_dim:])

        q = q.view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)

        if attn_mask is not None and attn_mask.dim() == 3 \
                and attn_mask.size(0) == batch * self.num_heads:
            attn_mask = attn_mask.view(batch, self.num_heads, query_len, key_len)

        if need_weights:
            scale = self.head_dim ** -0.5
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                attn_scores = attn_scores + attn_mask
            attn_weights = torch.softmax(attn_scores, dim=-1)
            if self.training and self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            raw_output = torch.matmul(attn_weights, v)
            weights_out = attn_weights.mean(dim=1)  # (batch, query_len, key_len)
        else:
            raw_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            weights_out = None

        attn_output = raw_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, query_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output, weights_out


class RotaryTransformerEncoderLayer(nn.TransformerEncoderLayer):
    '''Transformer encoder layer that swaps absolute positions for rotary attention.'''

    def __init__(self, *args, max_len: int=256, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.self_attn = RotaryMultiheadAttention(
            embed_dim=self.self_attn.embed_dim,
            num_heads=self.self_attn.num_heads,
            dropout=self.self_attn.dropout,
            bias=self.self_attn.in_proj_bias is not None,
            batch_first=self.self_attn.batch_first,
            max_len=max_len,
        )

    def forward(self, src: Tensor, src_mask: Optional[Tensor]=None,
                src_key_padding_mask: Optional[Tensor]=None,
                is_causal: bool=False) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
