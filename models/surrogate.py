"""Decoder-only conditional AR surrogate with a flow-matching patch head.

Backbone: token embed + theta prefix tokens + adaLN-Zero(theta) Transformer
(GQA, shared RoPE, fp32 RMSNorm).  Head: FlowMatchingHead over the full
token-feature vector, conditioned on the per-position embedding z_j plus a
learned horizon embedding E_k (one shared head for all K horizons; generation
uses k = 0).

Generation is incremental: the prefix + seed are prefilled once, then each
new patch costs ONE single-token backbone step (KV cache) plus `n_steps`
evaluations of the small flow head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from models.blocks import AdaLNZeroBlock, ParameterEncoder, RMSNorm, RotaryEmbedding
from models.flow_head import FlowMatchingHead


def normalize_theta(theta: torch.Tensor) -> torch.Tensor:
  """Map q in [1,8], chi in [-0.8, 0.8] to ~unit scale for conditioning."""
  out = theta.clone()
  out[..., 0] = 2.0 * (theta[..., 0] - 1.0) / 7.0 - 1.0
  out[..., 1:] = theta[..., 1:] / 0.8
  return out


class GWFlowSurrogateModel(nn.Module):
  def __init__(
    self,
    feat_dim: int,
    d_model: int = 256,
    n_layers: int = 12,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    n_prefix: int = 4,
    num_horizons: int = 4,
    head_width: int = 256,
    head_depth: int = 3,
    dropout: float = 0.0,
    grad_ckpt: bool = False,
    rope_max_len: int = 8192,
    use_abs_time: bool = True,
    max_positions: int = 4096,
  ):
    super().__init__()
    self.feat_dim = feat_dim
    self.d_model = d_model
    self.n_prefix = n_prefix
    self.num_horizons = num_horizons
    self.grad_ckpt = grad_ckpt

    self.token_embed = nn.Linear(feat_dim, d_model)
    # Absolute-time clock (§data: all waveforms share one peak-aligned grid, so
    # the patch index IS time-to-merger).  Without it the model must infer its
    # temporal location from the increment history alone, which is exactly the
    # feedback loop that produces early/late merger collapse in free rollout.
    self.abs_pos_emb = nn.Embedding(max_positions, d_model) if use_abs_time else None
    if self.abs_pos_emb is not None:
      nn.init.normal_(self.abs_pos_emb.weight, std=0.02)
    self.param_enc = ParameterEncoder(d_model, n_prefix)
    rope = RotaryEmbedding(d_model // n_heads, max_seq_len=rope_max_len)
    self.blocks = nn.ModuleList(
      AdaLNZeroBlock(d_model, n_heads, n_kv_heads, d_model, rope, dropout)
      for _ in range(n_layers)
    )
    self.final_norm = RMSNorm(d_model)

    self.horizon_emb = nn.Embedding(num_horizons, d_model)
    nn.init.normal_(self.horizon_emb.weight, std=0.02)
    self.head = FlowMatchingHead(feat_dim, cond_dim=d_model,
                                 width=head_width, depth=head_depth)

  # ---------------------------------------------------------------- #
  # Backbone                                                          #
  # ---------------------------------------------------------------- #
  def encode(
    self,
    token_features: torch.Tensor,          # (B, N, feat_dim)
    theta: torch.Tensor,                   # (B, 3)
    past_kv: list | None = None,
    use_cache: bool = False,
  ) -> tuple[torch.Tensor, list | None]:
    """Per-position context embeddings z.  With `past_kv` (incremental mode)
    the prefix is already in the cache and `token_features` must be a single
    new token."""
    prefix, cond = self.param_enc(normalize_theta(theta))
    x = self.token_embed(token_features)
    if self.abs_pos_emb is not None:
      start = 0 if past_kv is None else past_kv[0][0].shape[2] - self.n_prefix
      pos = torch.arange(start, start + x.shape[1], device=x.device)
      x = x + self.abs_pos_emb(pos)[None]
    if past_kv is None:
      x = torch.cat([prefix, x], dim=1)

    new_past: list | None = [] if use_cache else None
    for i, block in enumerate(self.blocks):
      layer_past = past_kv[i] if past_kv is not None else None
      if self.grad_ckpt and self.training and torch.is_grad_enabled():
        x, kv = checkpoint(block, x, cond, use_reentrant=False)
      else:
        x, kv = block(x, cond, past_kv=layer_past, use_cache=use_cache)
      if use_cache:
        new_past.append(kv)

    x = self.final_norm(x)
    if past_kv is None:
      x = x[:, self.n_prefix:]
    return x, new_past

  # ---------------------------------------------------------------- #
  # Generation                                                        #
  # ---------------------------------------------------------------- #
  @torch.no_grad()
  def generate(
    self,
    seed_features: torch.Tensor,           # (B, ctx, feat_dim)
    theta: torch.Tensor,                   # (B, 3)
    n_generate: int,
    n_steps: int = 20,
    n_samples: int = 1,
    temperature: float = 1.0,
    mean_of: int = 1,
  ) -> torch.Tensor:
    """Free-running autoregressive sampling (no teacher forcing).

    `temperature` scales the initial flow noise (0 => deterministic ODE from
    the origin).  `mean_of` > 1 averages that many head samples per step —
    a conditional-mean estimate that suppresses sampling jitter in the
    fed-back increments (the drift driver) at head-only cost.

    Returns (B, n_samples, n_generate, feat_dim) of NORMALIZED token features
    (same space as the training targets); split/un-normalize downstream.
    """
    self.eval()
    B = seed_features.shape[0]
    if n_samples > 1:
      seed_features = seed_features.repeat_interleave(n_samples, dim=0)
      theta = theta.repeat_interleave(n_samples, dim=0)

    h0 = self.horizon_emb.weight[0][None, :]
    z, past = self.encode(seed_features, theta, use_cache=True)
    cond = z[:, -1] + h0

    def draw(c: torch.Tensor) -> torch.Tensor:
      if mean_of <= 1:
        return self.head.sample(c, self.feat_dim, n_steps, temperature)
      rep = c.repeat_interleave(mean_of, dim=0)
      s = self.head.sample(rep, self.feat_dim, n_steps, temperature)
      return s.view(-1, mean_of, self.feat_dim).mean(dim=1)

    feats = []
    for i in range(n_generate):
      nxt = draw(cond)                                       # (B*S, feat_dim)
      feats.append(nxt)
      if i + 1 < n_generate:
        z, past = self.encode(nxt[:, None], theta, past_kv=past, use_cache=True)
        cond = z[:, -1] + h0

    gen = torch.stack(feats, dim=1)                          # (B*S, n_gen, D)
    return gen.view(B, n_samples, n_generate, self.feat_dim)

  @staticmethod
  def feature_dim(n_channels: int, patch_len: int) -> int:
    """Matches patchify.patches_to_token_features: log_a(1) + dphi(1) + res(P)."""
    return n_channels * (1 + 1 + patch_len)