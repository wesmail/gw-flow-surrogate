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

import warnings

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
    prefix: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, list | None]:
    """Per-position context embeddings z.  With `past_kv` (incremental mode)
    the prefix is already in the cache and `token_features` must be a single
    new token.

    Optional ``prefix`` / ``cond`` skip ``param_enc`` (used by ``generate_fast``);
    default ``None`` recomputes them from ``theta`` — identical to the old path.
    """
    if prefix is None or cond is None:
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

  @torch.no_grad()
  def generate_fast(
    self,
    seed_features: torch.Tensor,
    theta: torch.Tensor,
    n_generate: int,
    n_steps: int = 20,
    n_samples: int = 1,
    temperature: float = 1.0,
    mean_of: int = 1,
    noise: list[torch.Tensor] | None = None,
  ) -> torch.Tensor:
    """Opt-in fast path: same I/O as ``generate``, fewer CUDA launches.

    Precomputes ``param_enc`` once, pre-draws flow noise in the same RNG order
    as ``generate``, and decodes into a static KV cache.  Capture route
    (torch.compile CUDA-graphs, then manual CUDAGraph, then eager) is chosen
    once per ``(batch, n_steps, mean_of, cache_len)`` and cached on the module.

    Optional ``noise``: list of ``n_generate`` float32 tensors of shape
    ``(B*n_samples*max(mean_of,1), feat_dim)``.  If given, used as-is instead
    of drawing ``temperature * randn`` (shared-noise PE path).
    """
    self.eval()
    B0 = seed_features.shape[0]
    if n_samples > 1:
      seed_features = seed_features.repeat_interleave(n_samples, dim=0)
      theta = theta.repeat_interleave(n_samples, dim=0)
    B = seed_features.shape[0]
    ctx = int(seed_features.shape[1])
    device = seed_features.device
    dtype = seed_features.dtype
    D = self.feat_dim
    mo = max(int(mean_of), 1)
    noise_rows = B * mo

    if noise is None:
      # RNG order identical to generate: one randn(rows, D) per generated token
      noise = [
        temperature * torch.randn(noise_rows, D, device=device, dtype=torch.float32)
        for _ in range(n_generate)
      ]
    else:
      if len(noise) != n_generate:
        raise ValueError(f"noise length {len(noise)} != n_generate {n_generate}")
      noise = [
        n.to(device=device, dtype=torch.float32).expand(noise_rows, D).contiguous()
        if n.shape[0] == 1 and noise_rows > 1 else n.to(device=device, dtype=torch.float32)
        for n in noise
      ]
      for i, n in enumerate(noise):
        if n.shape != (noise_rows, D):
          raise ValueError(
            f"noise[{i}] shape {tuple(n.shape)} != ({noise_rows}, {D})"
          )

    prefix, cond_theta = self.param_enc(normalize_theta(theta))
    h0 = self.horizon_emb.weight[0][None, :]

    z, past = self.encode(
      seed_features, theta, use_cache=True, prefix=prefix, cond=cond_theta,
    )
    head_cond = z[:, -1] + h0
    if n_generate == 0:
      return seed_features.new_empty(B0, n_samples, 0, D)

    n_layers = len(self.blocks)
    pre_len = int(past[0][0].shape[2])
    cache_len = pre_len + max(n_generate - 1, 0)
    rope = self.blocks[0].attn.rope
    need = self.n_prefix + ctx + n_generate + 8
    if need > rope.max_seq_len:
      rope._build_cache(need)

    n_kv = past[0][0].shape[1]
    head_dim = past[0][0].shape[3]
    entry = self._get_fast_entry(
      B, D, n_steps, mo, cache_len, n_layers, n_kv, head_dim, dtype, device, h0,
    )
    k_bufs, v_bufs = entry["k_bufs"], entry["v_bufs"]
    cond_buf = entry["cond_buf"]
    attn_bias = entry["attn_bias"]
    pos_t = entry["pos"]
    tok_buf = entry["tok"]
    noise_buf = entry["noise"]
    abs_idx = entry["abs"]
    cond_buf.copy_(cond_theta)

    for li in range(n_layers):
      k_bufs[li].zero_()
      v_bufs[li].zero_()
      k_bufs[li][:, :, :pre_len].copy_(past[li][0])
      v_bufs[li][:, :, :pre_len].copy_(past[li][1])

    def _draw_from(c: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
      if mo <= 1:
        return self.head.sample_from(x0, c, n_steps)
      rep = c.repeat_interleave(mo, dim=0)
      s = self.head.sample_from(x0, rep, n_steps)
      return s.view(-1, mo, D).mean(dim=1)

    def _eager_step(tok_1: torch.Tensor, seq_pos: int, x0: torch.Tensor) -> torch.Tensor:
      x = self.token_embed(tok_1)
      if self.abs_pos_emb is not None:
        abs_idx.fill_(seq_pos - self.n_prefix)
        x = x + self.abs_pos_emb(abs_idx)[None]
      pos_t.fill_(seq_pos)
      self._fill_attn_bias(attn_bias, seq_pos)
      for li, block in enumerate(self.blocks):
        x = block.forward_static(
          x, cond_buf, k_bufs[li], v_bufs[li], pos_t, attn_bias,
        )
      z_new = self.final_norm(x)
      return _draw_from(z_new[:, -1] + h0, x0)

    feats = [_draw_from(head_cond, noise[0])]
    if n_generate == 1:
      return torch.stack(feats, dim=1).view(B0, n_samples, n_generate, D)

    route = entry["route"]
    for i in range(1, n_generate):
      seq_pos = pre_len + (i - 1)
      if route == "compile":
        tok_buf.copy_(feats[-1][:, None])
        noise_buf.copy_(noise[i])
        pos_t.fill_(seq_pos)
        abs_idx.fill_(seq_pos - self.n_prefix)
        self._fill_attn_bias(attn_bias, seq_pos)
        feats.append(entry["compiled"](tok_buf, noise_buf, pos_t, abs_idx, attn_bias))
      elif route == "cudagraph":
        st = entry["static"]
        st["tok"].copy_(feats[-1][:, None])
        st["noise"].copy_(noise[i])
        st["pos"].fill_(seq_pos)
        st["abs"].fill_(seq_pos - self.n_prefix)
        self._fill_attn_bias(st["bias"], seq_pos)
        entry["graph"].replay()
        feats.append(st["out"].clone())
      else:
        feats.append(_eager_step(feats[-1][:, None], seq_pos, noise[i]))

    return torch.stack(feats, dim=1).view(B0, n_samples, n_generate, D)

  @staticmethod
  def _fill_attn_bias(attn_bias: torch.Tensor, pos: int) -> None:
    """0 on slots ``<= pos``, -inf after (filled prefix for static attention)."""
    attn_bias.fill_(float("-inf"))
    attn_bias[..., : pos + 1] = 0.0

  def _set_static_full_buffer(self, enabled: bool) -> None:
    for block in self.blocks:
      block.attn._static_full_buffer = bool(enabled)

  def _get_fast_entry(
    self, B, D, n_steps, mo, cache_len, n_layers, n_kv, head_dim, dtype, device, h0,
  ):
    if not hasattr(self, "_fast_step_cache"):
      self._fast_step_cache = {}
    key = (B, n_steps, mo, cache_len, device.type, str(device))
    if key in self._fast_step_cache:
      return self._fast_step_cache[key]

    noise_rows = B * mo
    k_bufs = [
      torch.zeros(B, n_kv, cache_len, head_dim, device=device, dtype=dtype)
      for _ in range(n_layers)
    ]
    v_bufs = [
      torch.zeros(B, n_kv, cache_len, head_dim, device=device, dtype=dtype)
      for _ in range(n_layers)
    ]
    cond_buf = torch.zeros(B, self.d_model, device=device, dtype=dtype)
    attn_bias = torch.full(
      (1, 1, 1, cache_len), float("-inf"), device=device, dtype=dtype,
    )
    pos_t = torch.zeros(1, dtype=torch.long, device=device)
    tok_buf = torch.zeros(B, 1, D, device=device, dtype=dtype)
    noise_buf = torch.zeros(noise_rows, D, device=device, dtype=torch.float32)
    abs_idx = torch.zeros(1, dtype=torch.long, device=device)

    def _draw_from(c: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
      if mo <= 1:
        return self.head.sample_from(x0, c, n_steps)
      rep = c.repeat_interleave(mo, dim=0)
      s = self.head.sample_from(x0, rep, n_steps)
      return s.view(-1, mo, D).mean(dim=1)

    def core(tok_1, x0, pos, abs_i, bias):
      x = self.token_embed(tok_1)
      if self.abs_pos_emb is not None:
        x = x + self.abs_pos_emb(abs_i)[None]
      for li, block in enumerate(self.blocks):
        x = block.forward_static(
          x, cond_buf, k_bufs[li], v_bufs[li], pos, bias,
        )
      z = self.final_norm(x)
      return _draw_from(z[:, -1] + h0, x0)

    entry = dict(
      k_bufs=k_bufs, v_bufs=v_bufs, cond_buf=cond_buf, attn_bias=attn_bias,
      pos=pos_t, tok=tok_buf, noise=noise_buf, abs=abs_idx,
      route="eager", compiled=None, graph=None, static=None,
    )
    self._fast_capture_route = "eager"

    if device.type == "cuda":
      # Shape-static attention BEFORE tracing (a) or (b).  Cleared if both fail.
      self._set_static_full_buffer(True)
      warm_pos = max(min(cache_len // 2, cache_len - 1), 0)

      def _prepare_static_inputs(pos_val: int) -> None:
        pos_t.fill_(pos_val)
        abs_idx.fill_(max(pos_val - self.n_prefix, 0))
        self._fill_attn_bias(attn_bias, pos_val)

      # (a) torch.compile + CUDA graphs via reduce-overhead
      try:
        compiled = torch.compile(
          core, mode="reduce-overhead", fullgraph=True, dynamic=False,
        )
        _prepare_static_inputs(warm_pos)
        y0 = compiled(tok_buf, noise_buf, pos_t, abs_idx, attn_bias)
        _prepare_static_inputs(warm_pos)
        y1 = compiled(tok_buf, noise_buf, pos_t, abs_idx, attn_bias)
        assert y0.shape == (B, D) and y1.shape == (B, D)
        entry["route"] = "compile"
        entry["compiled"] = compiled
        self._fast_capture_route = "a:torch.compile(reduce-overhead)"
        self._fast_step_cache[key] = entry
        return entry
      except Exception as e:
        self._fast_compile_error = repr(e)
        warnings.warn(
          f"generate_fast route (a) torch.compile failed; trying (b). error={e!r}",
          stacklevel=2,
        )

      # (b) manual CUDAGraph; fusion compile errors show up on first CALL
      try:
        fused = core
        fuse_tag = ""
        try:
          fused_try = torch.compile(
            core, mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False,
          )
          _prepare_static_inputs(warm_pos)
          # errors often appear here, not at compile() construction
          _ = fused_try(tok_buf, noise_buf, pos_t, abs_idx, attn_bias)
          fused = fused_try
          fuse_tag = "+max-autotune-no-cudagraphs"
        except Exception as e_fuse:
          fused = core
          fuse_tag = ""
          warnings.warn(
            f"generate_fast route (b) fusion compile failed; "
            f"using plain CUDAGraph. error={e_fuse!r}",
            stacklevel=2,
          )

        static = {
          "tok": tok_buf, "noise": noise_buf, "pos": pos_t, "abs": abs_idx,
          "bias": attn_bias, "out": tok_buf.new_zeros(B, D),
        }
        _prepare_static_inputs(warm_pos)
        for _ in range(3):
          _prepare_static_inputs(warm_pos)
          static["out"].copy_(fused(
            static["tok"], static["noise"], static["pos"],
            static["abs"], static["bias"],
          ))
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
          static["out"].copy_(fused(
            static["tok"], static["noise"], static["pos"],
            static["abs"], static["bias"],
          ))
        entry["route"] = "cudagraph"
        entry["graph"] = g
        entry["static"] = static
        self._fast_capture_route = f"b:CUDAGraph{fuse_tag}"
        self._fast_step_cache[key] = entry
        return entry
      except Exception as e:
        self._fast_cudagraph_error = repr(e)
        warnings.warn(
          f"generate_fast route (b) CUDAGraph failed; falling back to eager. "
          f"error={e!r}",
          stacklevel=2,
        )
        self._set_static_full_buffer(False)

    self._fast_step_cache[key] = entry
    return entry

  @staticmethod
  def feature_dim(n_channels: int, patch_len: int) -> int:
    """Matches patchify.patches_to_token_features: log_a(1) + dphi(1) + res(P)."""
    return n_channels * (1 + 1 + patch_len)