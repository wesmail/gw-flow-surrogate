"""Backbone blocks for the flow-head surrogate.

Merged from the adaLN surrogate (theta conditioning) and the MAR forecaster
(engineering): fp32 RMSNorm, ONE shared RoPE module for the whole stack,
grouped-query attention (smaller KV projections + KV cache), and incremental
decoding support so `generate` costs O(n) backbone token evaluations instead
of O(n^2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
  """RMSNorm computed in fp32 (safe under bf16-mixed), cast back to input dtype."""

  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
    return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
  def __init__(self, dim: int, hidden_mult: float = 4.0):
    super().__init__()
    hidden = int(dim * hidden_mult)
    self.w1 = nn.Linear(dim, hidden, bias=False)
    self.w2 = nn.Linear(dim, hidden, bias=False)
    self.w3 = nn.Linear(hidden, dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w3(F.silu(self.w1(x)) * self.w2(x))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
  x1, x2 = x.chunk(2, dim=-1)
  return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
  """Shared RoPE cache.  One instance serves every attention layer (the old
  code built one per layer, each with its own cos/sin buffers).

  `slice(start, end)` returns cos/sin for ABSOLUTE positions [start, end),
  which is what incremental decoding with a KV cache needs.
  """

  def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10_000.0):
    super().__init__()
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    self.max_seq_len = 0
    self._build_cache(max_seq_len)

  def _build_cache(self, seq_len: int) -> None:
    t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, self.inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    self.register_buffer("cos_cached", emb.cos()[None, None], persistent=False)
    self.register_buffer("sin_cached", emb.sin()[None, None], persistent=False)
    self.max_seq_len = seq_len

  def slice(self, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
    if end > self.max_seq_len:
      self._build_cache(max(end, 2 * self.max_seq_len))
    return (
      self.cos_cached[:, :, start:end],
      self.sin_cached[:, :, start:end],
    )


class CausalSelfAttention(nn.Module):
  """Grouped-query attention with flash SDPA, RoPE, and an optional KV cache.

  The cache stores keys/values at n_kv_heads resolution (rep-expansion to the
  query head count happens after concatenation), so cache memory is
  n_kv_heads / n_heads of a full MHA cache.
  """

  def __init__(
    self,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    rope: RotaryEmbedding,
    dropout: float = 0.0,
  ):
    super().__init__()
    assert dim % n_heads == 0 and n_heads % n_kv_heads == 0
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads
    self.head_dim = dim // n_heads
    self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
    self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
    self.out = nn.Linear(dim, dim, bias=False)
    self.dropout = dropout
    self.rope = rope

  def forward(
    self,
    x: torch.Tensor,
    past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    use_cache: bool = False,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    b, n, _ = x.shape
    q = self.q_proj(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, n, self.n_kv_heads, self.head_dim).transpose(1, 2)

    pos0 = past_kv[0].shape[2] if past_kv is not None else 0
    if past_kv is not None and n != 1:
      raise ValueError("Incremental decoding expects exactly one new token.")
    cos, sin = self.rope.slice(pos0, pos0 + n)
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    if past_kv is not None:
      k = torch.cat([past_kv[0], k], dim=2)
      v = torch.cat([past_kv[1], v], dim=2)
    new_kv = (k, v) if use_cache else None

    rep = self.n_heads // self.n_kv_heads
    if rep > 1:
      k = k.repeat_interleave(rep, dim=1)
      v = v.repeat_interleave(rep, dim=1)

    out = F.scaled_dot_product_attention(
      q, k, v,
      dropout_p=self.dropout if self.training else 0.0,
      # Prefill: full causal mask.  Incremental (single query): attend to the
      # entire cache, which IS the causal pattern for the last position.
      is_causal=past_kv is None,
    )
    out = out.transpose(1, 2).reshape(b, n, -1)
    return self.out(out), new_kv


class AdaLNZeroBlock(nn.Module):
  """Pre-norm block with DiT-style adaLN-Zero conditioning on theta.

  The modulation is per-sample and position-independent, so it commutes with
  KV caching.
  """

  def __init__(
    self,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    cond_dim: int,
    rope: RotaryEmbedding,
    dropout: float = 0.0,
  ):
    super().__init__()
    self.norm1 = RMSNorm(dim)
    self.attn = CausalSelfAttention(dim, n_heads, n_kv_heads, rope, dropout)
    self.norm2 = RMSNorm(dim)
    self.mlp = SwiGLU(dim)
    self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))
    nn.init.zeros_(self.ada[-1].weight)
    nn.init.zeros_(self.ada[-1].bias)

  def forward(
    self,
    x: torch.Tensor,
    cond: torch.Tensor,
    past_kv: tuple | None = None,
    use_cache: bool = False,
  ) -> tuple[torch.Tensor, tuple | None]:
    shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)
    h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
    a, new_kv = self.attn(h, past_kv=past_kv, use_cache=use_cache)
    x = x + gate1.unsqueeze(1) * a
    h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
    x = x + gate2.unsqueeze(1) * self.mlp(h)
    return x, new_kv


class ParameterEncoder(nn.Module):
  """Embed theta=(q, chi1z, chi2z) into prefix tokens + global cond vector."""

  def __init__(self, cond_dim: int, n_prefix: int = 4):
    super().__init__()
    self.n_prefix = n_prefix
    self.mlp = nn.Sequential(
      nn.Linear(3, cond_dim),
      nn.SiLU(),
      nn.Linear(cond_dim, cond_dim),
      nn.SiLU(),
    )
    self.prefix_proj = nn.Linear(cond_dim, n_prefix * cond_dim)

  def forward(self, theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = self.mlp(theta)
    prefix = self.prefix_proj(h).view(theta.shape[0], self.n_prefix, -1)
    return prefix, h