"""Correctness smoke test for the hybrid flow surrogate."""
import math
import torch

torch.manual_seed(0)

from models.surrogate import GWFlowSurrogateModel
from datasets.patchify import patchify_tracks_batched, build_token_features_batched
from models.lightning_module import GWFlowSurrogateLit

B, C, P, N = 3, 1, 8, 24
T = N * P + 1
D = GWFlowSurrogateModel.feature_dim(C, P)

model = GWFlowSurrogateModel(feat_dim=D, d_model=64, n_layers=3, n_heads=4,
                             n_kv_heads=2, n_prefix=2, num_horizons=4,
                             head_width=64, head_depth=2)
model.eval()

feat = torch.randn(B, N, D)
theta = torch.stack([torch.rand(B) * 7 + 1, torch.rand(B) * 1.6 - 0.8,
                     torch.rand(B) * 1.6 - 0.8], dim=-1)

# ---- 1. KV cache == full recompute -------------------------------------- #
with torch.no_grad():
  z_full, _ = model.encode(feat, theta)                       # (B, N, d)
  ctx = 5
  z_pre, past = model.encode(feat[:, :ctx], theta, use_cache=True)
  zs = [z_pre]
  for j in range(ctx, N):
    z_j, past = model.encode(feat[:, j:j + 1], theta, past_kv=past, use_cache=True)
    zs.append(z_j)
  z_inc = torch.cat(zs, dim=1)
err = (z_full - z_inc).abs().max().item()
print(f"[1] KV-cache max abs err vs full recompute: {err:.3e}")
assert err < 1e-5

# ---- 2. feature split roundtrip ----------------------------------------- #
la = torch.randn(B, N, C); dp = torch.randn(B, N, C); res = torch.randn(B, N, C, P)
f = build_token_features_batched(la, dp, res)
la2, dp2, res2 = GWFlowSurrogateLit._split_features(f, C, P)
assert torch.equal(la, la2) and torch.equal(dp, dp2) and torch.equal(res, res2)
print("[2] token-feature split roundtrip: exact")

# ---- 3. vectorized reconstruction == loop reference ---------------------- #
lit = GWFlowSurrogateLit(d_model=64, n_layers=3, n_heads=4, n_kv_heads=2,
                         n_channels=C, patch_len=P, num_horizons=4,
                         head_width=64, head_depth=2, residual_scale=0.37)
la_n = torch.randn(B, N); dp_n = torch.randn(B, N); res_n = torch.randn(B, N, P)
stats = [torch.randn(B).abs() + 0.5 for _ in range(4)]

def loop_reference(la_n, dp_n, res_n, la_m, la_s, dp_m, dp_s, P, scale):
  la_phys = la_n * la_s[:, None] + la_m[:, None]
  dphi = dp_n * dp_s[:, None] + dp_m[:, None]
  res = res_n * scale
  s = torch.arange(P).float() / P
  bnd = torch.zeros(B)
  chunks = []
  for k in range(N):
    phase = bnd[:, None] + s[None] * dphi[:, k:k + 1] + res[:, k]
    chunks.append(torch.exp(la_phys[:, k:k + 1]) * torch.exp(-1j * phase))
    bnd = bnd + dphi[:, k]
  return torch.cat(chunks, dim=1)

h_vec = lit._reconstruct_waveform(la_n, dp_n, res_n, *stats, P)
h_ref = loop_reference(la_n, dp_n, res_n, *stats, P, 0.37)
err = (h_vec - h_ref).abs().max().item()
print(f"[3] vectorized reconstruction max abs err: {err:.3e}")
assert err < 2e-4

# ---- 4. synthetic batch: train step + val rollout ------------------------ #
def make_batch():
  t = torch.linspace(0, 1, T)
  la_track = (-2 + 1.5 * t)[None, None].repeat(B, C, 1) + 0.01 * torch.randn(B, C, T)
  phi_track = (40 * t ** 2 + 10 * t)[None, None].repeat(B, C, 1)
  return {
    "log_a_track": la_track, "phi_track": phi_track, "theta": theta.clone(),
    "ref_phase": torch.zeros(B),
    "context_patches": torch.full((B,), 4, dtype=torch.long),
    "target_start": torch.full((B,), 4, dtype=torch.long),
    "norm_log_a_mean": torch.zeros(B, C), "norm_log_a_std": torch.ones(B, C),
    "norm_dphi_mean": torch.zeros(B, C), "norm_dphi_std": torch.ones(B, C),
  }

lit.train()
loss = lit._flow_loss(make_batch(), "train")
loss.backward()
g = sum(p.grad.abs().sum().item() for p in lit.parameters() if p.grad is not None)
print(f"[4] train step: loss={loss.item():.4f}, grad-abs-sum={g:.2f} (finite: "
      f"{math.isfinite(loss.item()) and math.isfinite(g)})")
assert math.isfinite(loss.item()) and g > 0

lit.eval()
mm = lit._rollout_mismatch(make_batch())
print(f"[5] rollout mismatch on untrained model: {mm.item():.4f} (in [0, 2])")
assert 0.0 <= mm.item() <= 2.0

# ---- 5. loss_on_context=False path -------------------------------------- #
lit2 = GWFlowSurrogateLit(d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                          n_channels=C, patch_len=P, num_horizons=4,
                          head_width=64, head_depth=2, loss_on_context=False)
l2 = lit2._flow_loss(make_batch(), "train")
print(f"[6] loss_on_context=False: loss={l2.item():.4f}")

# ---- 6. rollout speed: KV cache vs full re-encode ------------------------ #
import time
model.eval()
seed = feat[:, :4]
with torch.no_grad():
  t0 = time.perf_counter()
  for _ in range(3):
    model.generate(seed, theta, n_generate=60, n_steps=10)
  t_cache = (time.perf_counter() - t0) / 3

  def gen_nocache(seed, theta, n_gen, n_steps):
    x = seed
    h0 = model.horizon_emb.weight[0][None]
    outs = []
    for _ in range(n_gen):
      z, _ = model.encode(x, theta)
      nxt = model.head.sample(z[:, -1] + h0, D, n_steps)
      outs.append(nxt)
      x = torch.cat([x, nxt[:, None]], dim=1)
    return torch.stack(outs, 1)

  t0 = time.perf_counter()
  for _ in range(3):
    gen_nocache(seed, theta, 60, 10)
  t_full = (time.perf_counter() - t0) / 3
print(f"[7] 60-patch rollout: KV cache {t_cache*1e3:.0f} ms vs full re-encode "
      f"{t_full*1e3:.0f} ms  ({t_full/t_cache:.1f}x)")

print("\nALL CHECKS PASSED")
