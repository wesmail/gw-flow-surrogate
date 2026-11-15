"""PyTorch Lightning module: flow-matching AR surrogate training.

What changed vs. the parametric-head module:
  * Loss is rectified-flow velocity MSE on the full token-feature vector
    (joint distribution, no Student-t/Gaussian/von Mises factorization).
  * ALL horizons are trained in ONE flow-head forward: (z_j + E_k, target)
    pairs from every horizon are flattened and concatenated first, so K
    horizons cost K x more rows through a tiny MLP, not K separate passes.
  * Scheduled sampling is gone — the stochastic head + mismatch-monitored
    checkpointing replace it (it was also a no-op at context_patches=1).
  * Optimizer: no weight decay on norms/embeddings/biases; step-based linear
    warmup + cosine to min_lr_ratio.
  * Validation still reconstructs h = A e^{-i phi} in physical units and logs
    the phase-maximized mismatch — this remains the checkpoint monitor.
    Rollout now uses the KV cache: O(n) backbone token evaluations.

Data pipeline (datamodule / dataset / patchify) is UNCHANGED except that this
module additionally scales phi_residual by 1/residual_scale after patchify so
all head dimensions are O(1) for the N(0, I) flow prior, and un-scales at
reconstruction.  Set residual_scale to roughly the residual std of your
training set (see compute_residual_scale below).
"""

from __future__ import annotations

import math

import torch
from lightning.pytorch import LightningModule

from datasets.patchify import build_token_features_batched, patchify_tracks_batched
from models.amp_interp import AMP_MODES, log_amp_per_sample
from models.surrogate import GWFlowSurrogateModel


class GWFlowSurrogateLit(LightningModule):
  def __init__(
    self,
    # backbone
    d_model: int = 256,
    n_layers: int = 12,
    n_heads: int = 8,
    n_kv_heads: int = 4,
    n_channels: int = 1,
    patch_len: int = 32,
    n_prefix: int = 4,
    dropout: float = 0.0,
    grad_ckpt: bool = False,
    # flow head
    head_width: int = 256,
    head_depth: int = 3,
    num_horizons: int = 4,
    n_sampling_steps: int = 20,
    # loss
    loss_on_context: bool = True,
    residual_scale: float = 1.0,
    # optimization
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
    warmup_steps: int = 500,
    max_steps: int = -1,            # -1 => trainer.estimated_stepping_batches
    min_lr_ratio: float = 0.1,
    # validation rollout
    rollout_n_waveforms: int = 4,
    rollout_n_samples: int = 1,
    rollout_max_patches: int = 0,   # 0 => full future region
    rollout_temperature: float = 1.0,
    rollout_mean_of: int = 1,
    use_abs_time: bool = True,
    amp_mode: str = "constant",
    la_weight: float = 1.0,         # loss weight of the log-amplitude feature
    dphi_weight: float = 1.0,       # loss weight of the phase-advance feature
  ) -> None:
    super().__init__()
    if amp_mode not in AMP_MODES:
      raise ValueError(f"amp_mode must be one of {AMP_MODES}, got {amp_mode!r}")
    self.save_hyperparameters()

    feat_dim = GWFlowSurrogateModel.feature_dim(n_channels, patch_len)
    self.model = GWFlowSurrogateModel(
      feat_dim=feat_dim,
      d_model=d_model,
      n_layers=n_layers,
      n_heads=n_heads,
      n_kv_heads=n_kv_heads,
      n_prefix=n_prefix,
      num_horizons=num_horizons,
      head_width=head_width,
      head_depth=head_depth,
      dropout=dropout,
      grad_ckpt=grad_ckpt,
      use_abs_time=use_abs_time,
    )

  # ------------------------------------------------------------------ #
  # GPU patchify (unchanged pipeline + residual scaling)                #
  # ------------------------------------------------------------------ #
  def _patchify_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if "token_features" in batch:
      return batch

    patches = patchify_tracks_batched(
      batch["log_a_track"], batch["phi_track"], self.hparams.patch_len
    )
    la_m = batch["norm_log_a_mean"].unsqueeze(1)
    la_s = batch["norm_log_a_std"].unsqueeze(1)
    dp_m = batch["norm_dphi_mean"].unsqueeze(1)
    dp_s = batch["norm_dphi_std"].unsqueeze(1)
    patches["log_a"] = (patches["log_a"] - la_m) / la_s
    patches["delta_phi"] = (patches["delta_phi"] - dp_m) / dp_s
    patches["phi_residual"] = patches["phi_residual"] / self.hparams.residual_scale

    batch["token_features"] = build_token_features_batched(
      patches["log_a"], patches["delta_phi"], patches["phi_residual"]
    )
    batch["log_a"] = patches["log_a"]
    batch["delta_phi"] = patches["delta_phi"]
    batch["phi_residual"] = patches["phi_residual"]
    batch["mask"] = torch.ones(
      patches["log_a"].shape[0], patches["log_a"].shape[1],
      dtype=torch.bool, device=patches["log_a"].device,
    )
    return batch

  def on_after_batch_transfer(self, batch, dataloader_idx: int = 0):
    return self._patchify_batch(batch)

  # ------------------------------------------------------------------ #
  # Flow-matching loss (single batched head call over all horizons)     #
  # ------------------------------------------------------------------ #
  # ------------------------------------------------------------------ #
  # Per-feature loss weights                                            #
  # ------------------------------------------------------------------ #
  def _feature_weights(self, D: int, device, dtype):
    """Weight per token feature, or None when every weight is 1.

    Token layout per channel c (channel-major, see _split_features):
      c*(2+P)+0 : log-amplitude (patch mean)
      c*(2+P)+1 : phase advance delta-phi
      c*(2+P)+2 .. c*(2+P)+P+1 : phase residuals
    The weighted loss is normalised by the mean weight, so its size stays
    comparable to the unweighted loss; only the balance between features
    changes.  In flow matching the per-feature optimum (the conditional mean
    velocity) does not depend on these weights; they only decide which errors
    the network reduces first when it cannot reduce all of them.
    """
    la_w = float(getattr(self.hparams, "la_weight", 1.0))
    dp_w = float(getattr(self.hparams, "dphi_weight", 1.0))
    if la_w == 1.0 and dp_w == 1.0:
      return None
    C, P = int(self.hparams.n_channels), int(self.hparams.patch_len)
    if D != C * (2 + P):
      raise ValueError(f"feature dim {D} != n_channels*(2+patch_len) = {C * (2 + P)}")
    w = torch.ones(D, device=device, dtype=dtype)
    for c in range(C):
      w[c * (2 + P) + 0] = la_w
      w[c * (2 + P) + 1] = dp_w
    return w

  def _log_feature_losses(self, sq: torch.Tensor, stage: str, batch_size: int) -> None:
    """Unweighted mean squared velocity error per feature group (for monitoring)."""
    C, P = int(self.hparams.n_channels), int(self.hparams.patch_len)
    parts = sq.view(sq.shape[0], C, 2 + P)
    on_step = stage == "train"
    self.log(f"{stage}_loss_la", parts[..., 0].mean(), on_step=on_step, on_epoch=True,
             batch_size=batch_size)
    self.log(f"{stage}_loss_dphi", parts[..., 1].mean(), on_step=on_step, on_epoch=True,
             batch_size=batch_size)
    self.log(f"{stage}_loss_res", parts[..., 2:].mean(), on_step=on_step, on_epoch=True,
             batch_size=batch_size)

  def _flow_loss(self, batch: dict[str, torch.Tensor], stage: str) -> torch.Tensor:
    batch = self._patchify_batch(batch)
    feat = batch["token_features"]                       # (B, N, D)
    z, _ = self.model.encode(feat, batch["theta"])       # (B, N, d_model)

    N = feat.shape[1]
    ctx = int(batch["context_patches"][0].item())
    K = self.hparams.num_horizons

    conds, tgts = [], []
    for k in range(K):
      j_max = N - 2 - k                                  # last position with a target
      j0 = 0 if self.hparams.loss_on_context else max(0, ctx - 1 - k)
      if j_max < j0:
        continue
      zk = z[:, j0:j_max + 1] + self.model.horizon_emb.weight[k]
      tk = feat[:, j0 + 1 + k:N]                         # targets j+1+k, same count
      conds.append(zk.reshape(-1, zk.shape[-1]))
      tgts.append(tk.reshape(-1, tk.shape[-1]))

    cond = torch.cat(conds, dim=0)                       # (M, d_model)
    tgt = torch.cat(tgts, dim=0)                         # (M, D)

    tau = torch.rand(tgt.shape[0], device=tgt.device, dtype=tgt.dtype)
    x0 = torch.randn_like(tgt)
    x_tau = (1 - tau[:, None]) * x0 + tau[:, None] * tgt
    v_star = tgt - x0
    v_pred = self.model.head(x_tau, tau, cond)
    sq = (v_pred - v_star).pow(2)                        # (M, D)
    w = self._feature_weights(sq.shape[-1], sq.device, sq.dtype)
    if w is None:                                        # all weights 1: old loss, unchanged
      loss = sq.mean()
    else:
      loss = (sq * w).sum(-1).mean() / w.mean() / sq.shape[-1]
    self._log_feature_losses(sq.detach(), stage, feat.shape[0])

    self.log(f"{stage}_loss", loss, prog_bar=True,
             on_step=(stage == "train"), on_epoch=True,
             batch_size=feat.shape[0])
    return loss

  # ------------------------------------------------------------------ #
  # Physical reconstruction + rollout mismatch (checkpoint monitor)     #
  # ------------------------------------------------------------------ #
  @staticmethod
  def _split_features(feat: torch.Tensor, n_channels: int, patch_len: int):
    """(..., N, C*(2+P)) -> log_a (..., N, C), dphi (..., N, C), res (..., N, C, P).
    Inverse of build_token_features_batched's channel-major layout."""
    parts = feat.view(*feat.shape[:-1], n_channels, 2 + patch_len)
    return parts[..., 0], parts[..., 1], parts[..., 2:]

  @staticmethod
  def _norm_broadcast(stat: torch.Tensor) -> torch.Tensor:
    return stat[:, None] if stat.ndim == 1 else stat[:, :1]

  def _reconstruct_waveform(
    self,
    la_n: torch.Tensor,      # (B, n)  normalized, channel 0
    dphi_n: torch.Tensor,    # (B, n)
    res_n: torch.Tensor,     # (B, n, P)  scaled by 1/residual_scale
    la_m, la_s, dp_m, dp_s,
    patch_len: int,
  ) -> torch.Tensor:
    """h = A e^{-i phi} at sample resolution, physical units, fp32.

    Phase: exact by construction (patch-start phase + linear ramp + residual).
    Amplitude: patch-mean log_a expanded by ``hparams.amp_mode``
    (``constant`` / ``linear`` / ``pchip``); see ``models.amp_interp``.
    """
    la_n, dphi_n = la_n.float(), dphi_n.float()
    res = res_n.float() * self.hparams.residual_scale
    la_m = self._norm_broadcast(la_m); la_s = self._norm_broadcast(la_s)
    dp_m = self._norm_broadcast(dp_m); dp_s = self._norm_broadcast(dp_s)
    la_phys = la_n * la_s + la_m
    dphi_rad = dphi_n * dp_s + dp_m
    B, n = la_phys.shape
    s = torch.arange(patch_len, device=la_phys.device, dtype=la_phys.dtype) / patch_len
    # Vectorized over patches: boundary phase by cumulative sum (fp32).
    bnd = torch.cumsum(dphi_rad, dim=1) - dphi_rad                     # (B, n) phase at patch start
    phase = bnd[:, :, None] + s[None, None, :] * dphi_rad[:, :, None] + res
    amp_mode = getattr(self.hparams, "amp_mode", "constant")
    la_t = log_amp_per_sample(la_phys, patch_len, amp_mode)           # (B, n*P)
    amp = torch.exp(la_t).view(B, n, patch_len)
    return (amp * torch.exp(-1j * phase)).reshape(B, n * patch_len)

  @torch.no_grad()
  def _rollout_mismatch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    batch = self._patchify_batch(batch)
    if "norm_log_a_mean" not in batch:
      return batch["token_features"].new_tensor(1.0)
    P = self.hparams.patch_len
    C = self.hparams.n_channels
    ctx = max(int(batch["context_patches"][0].item()), 1)
    W = min(self.hparams.rollout_n_waveforms, batch["token_features"].shape[0])
    S = self.hparams.rollout_n_samples
    n_avail = int(batch["mask"][:W].sum(dim=1).min().item())
    n_gen = n_avail - ctx
    if self.hparams.rollout_max_patches > 0:
      n_gen = min(n_gen, self.hparams.rollout_max_patches)
    if n_gen <= 0:
      return batch["token_features"].new_tensor(1.0)

    seed = batch["token_features"][:W, :ctx]
    gen = self.model.generate(
      seed, batch["theta"][:W], n_gen,
      n_steps=self.hparams.n_sampling_steps, n_samples=S,
      temperature=self.hparams.rollout_temperature,
      mean_of=self.hparams.rollout_mean_of,
    )                                                    # (W, S, n_gen, D)
    la_g, dphi_g, res_g = self._split_features(gen, C, P)

    la_m = batch["norm_log_a_mean"][:W]; la_s = batch["norm_log_a_std"][:W]
    dp_m = batch["norm_dphi_mean"][:W]; dp_s = batch["norm_dphi_std"][:W]

    # Full sequence then slice generated samples so amp interp sees left neighbours.
    la_t_full = batch["log_a"][:W, : ctx + n_gen, 0]
    dp_t_full = batch["delta_phi"][:W, : ctx + n_gen, 0]
    res_t_full = batch["phi_residual"][:W, : ctx + n_gen, 0]
    h_true = self._reconstruct_waveform(
      la_t_full, dp_t_full, res_t_full, la_m, la_s, dp_m, dp_s, P,
    )[:, ctx * P :]

    la_seed = batch["log_a"][:W, :ctx, 0]
    dp_seed = batch["delta_phi"][:W, :ctx, 0]
    res_seed = batch["phi_residual"][:W, :ctx, 0]
    mismatches = []
    for s_idx in range(S):
      la_full = torch.cat([la_seed, la_g[:, s_idx, :, 0]], dim=1)
      dp_full = torch.cat([dp_seed, dphi_g[:, s_idx, :, 0]], dim=1)
      res_full = torch.cat([res_seed, res_g[:, s_idx, :, 0]], dim=1)
      h_pred = self._reconstruct_waveform(
        la_full, dp_full, res_full, la_m, la_s, dp_m, dp_s, P,
      )[:, ctx * P :]
      num = torch.abs((h_pred * h_true.conj()).sum(dim=1))
      den = torch.sqrt(
        (h_pred.abs() ** 2).sum(1) * (h_true.abs() ** 2).sum(1)
      ).clamp_min(1e-30)
      mismatches.append(1.0 - num / den)                 # (W,)
    mm = torch.stack(mismatches, dim=1)                  # (W, S)
    if S > 1:
      self.log("val_mismatch_best", mm.min(dim=1).values.mean(),
               on_step=False, on_epoch=True)
    return mm.mean()

  # ------------------------------------------------------------------ #
  # Lightning hooks                                                     #
  # ------------------------------------------------------------------ #
  def training_step(self, batch, batch_idx: int) -> torch.Tensor:
    return self._flow_loss(batch, "train")

  def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
    loss = self._flow_loss(batch, "val")
    if batch_idx == 0:
      self.log("val_mismatch", self._rollout_mismatch(batch),
               prog_bar=True, on_step=False, on_epoch=True)
    return loss

  def test_step(self, batch, batch_idx: int) -> torch.Tensor:
    if batch_idx == 0:
      self.log("test_mismatch", self._rollout_mismatch(batch),
               on_step=False, on_epoch=True)
    return self._flow_loss(batch, "test")

  def configure_optimizers(self):
    hp = self.hparams
    decay, no_decay = [], []
    for name, p in self.named_parameters():
      if not p.requires_grad:
        continue
      (no_decay if p.ndim < 2 or "emb" in name or "norm" in name else decay).append(p)
    opt = torch.optim.AdamW(
      [{"params": decay, "weight_decay": hp.weight_decay},
       {"params": no_decay, "weight_decay": 0.0}],
      lr=hp.learning_rate, betas=(hp.beta1, hp.beta2),
    )
    max_steps = hp.max_steps
    if max_steps is None or max_steps <= 0:
      max_steps = int(self.trainer.estimated_stepping_batches)

    def lr_lambda(step: int) -> float:
      if step < hp.warmup_steps:
        return step / max(1, hp.warmup_steps)
      t = (step - hp.warmup_steps) / max(1, max_steps - hp.warmup_steps)
      cos = 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
      return hp.min_lr_ratio + (1 - hp.min_lr_ratio) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return {"optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"}}


def compute_residual_scale(datamodule, patch_len: int, n_batches: int = 8) -> float:
  """One-off helper: std of the de-ramped phase residual over a few train
  batches.  Set model.residual_scale to this value in the config."""
  from torch.utils.data import DataLoader

  from datasets.datamodule import _collate_batch

  datamodule.setup("fit")
  ds = datamodule.train_ds
  # Shuffled train_dataloader thrash-loads many full shards; for this probe
  # walk the split in order so only the first shard(s) are touched.
  if not ds.preload and not ds.max_cached_shards:
    ds.max_cached_shards = 2
  loader = DataLoader(
    ds,
    batch_size=int(datamodule.hparams.batch_size),
    shuffle=False,
    num_workers=0,
    collate_fn=_collate_batch,
  )
  sq, n = 0.0, 0
  for i, b in enumerate(loader):
    p = patchify_tracks_batched(b["log_a_track"], b["phi_track"], patch_len)
    r = p["phi_residual"]
    sq += float((r ** 2).sum()); n += r.numel()
    if i + 1 >= n_batches:
      break
  return max((sq / max(n, 1)) ** 0.5, 1e-8)