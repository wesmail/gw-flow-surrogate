"""Tests for the per-feature loss weights (run: pytest tests/test_loss_weights.py -v)."""
import torch

from models.lightning_module import GWFlowSurrogateLit

P = 32


def _model(**kw):
    torch.manual_seed(0)
    return GWFlowSurrogateLit(d_model=32, n_layers=2, n_heads=2, n_kv_heads=1, head_width=32,
                              head_depth=1, patch_len=P, **kw)


def test_defaults_are_old_behaviour():
    m = _model()
    assert m.hparams.la_weight == 1.0 and m.hparams.dphi_weight == 1.0
    assert m._feature_weights(2 + P, "cpu", torch.float32) is None


def test_weight_vector_layout():
    m = _model(la_weight=10.0, dphi_weight=30.0)
    w = m._feature_weights(2 + P, "cpu", torch.float32)
    assert w[0] == 10 and w[1] == 30 and torch.all(w[2:] == 1)


def test_weighted_loss_formula():
    """With weights, an error only in delta-phi counts dphi_weight times more."""
    m = _model(la_weight=1.0, dphi_weight=30.0)
    D = 2 + P
    w = m._feature_weights(D, "cpu", torch.float32)
    sq = torch.zeros(4, D); sq[:, 1] = 1.0            # error only in delta-phi
    loss_w = (sq * w).sum(-1).mean() / w.mean() / D
    sq2 = torch.zeros(4, D); sq2[:, 5] = 1.0          # same error, in one residual
    loss_r = (sq2 * w).sum(-1).mean() / w.mean() / D
    assert torch.isclose(loss_w / loss_r, torch.tensor(30.0))
    # all-ones weights would give the plain mean
    ones = torch.ones(D)
    assert torch.isclose((sq * ones).sum(-1).mean() / ones.mean() / D, sq.mean())


def test_old_checkpoint_loads(tmp_path):
    import lightning
    m = _model()
    hp = {k: v for k, v in dict(m.hparams).items() if k not in ("la_weight", "dphi_weight")}
    ck = tmp_path / "old.ckpt"
    torch.save({"state_dict": m.state_dict(), "hyper_parameters": hp,
                "pytorch-lightning_version": lightning.__version__}, ck)
    m2 = GWFlowSurrogateLit.load_from_checkpoint(str(ck), map_location="cpu")
    assert m2.hparams.la_weight == 1.0 and m2.hparams.dphi_weight == 1.0
