"""Amplitude reconstruction from per-patch mean log_a.

Tokens store one log-amplitude per patch (the mean over that patch).  This
module expands ``(B, n)`` patch values to ``(B, n*P)`` sample-level log_a with:

* ``constant`` — flat step (legacy / default)
* ``linear``   — linear interp between patch centres, flat outside
* ``pchip``    — monotone cubic (Fritsch–Carlson / SciPy PCHIP), flat outside

Centres sit at sample index ``k*P + (P-1)/2``.  Outside ``[xc[0], xc[-1]]``
the amplitude is held constant (no extrapolation).
"""

from __future__ import annotations

import torch

AMP_MODES = ("constant", "linear", "pchip")


def log_amp_per_sample(
    la_patch: torch.Tensor,
    patch_len: int,
    mode: str = "constant",
) -> torch.Tensor:
    """Expand patch-mean log_a ``(B, n)`` → per-sample log_a ``(B, n*patch_len)``."""
    if mode not in AMP_MODES:
        raise ValueError(f"amp_mode must be one of {AMP_MODES}, got {mode!r}")
    if la_patch.ndim != 2:
        raise ValueError(f"la_patch must be (B, n), got shape {tuple(la_patch.shape)}")
    B, n = la_patch.shape
    P = int(patch_len)
    if P < 1:
        raise ValueError(f"patch_len must be >= 1, got {P}")
    if mode == "constant" or n < 2:
        return la_patch.repeat_interleave(P, dim=1)

    device, dtype = la_patch.device, la_patch.dtype
    x = torch.arange(n * P, device=device, dtype=dtype)
    xc = torch.arange(n, device=device, dtype=dtype) * P + (P - 1) / 2.0
    xq = x.clamp(xc[0], xc[-1])

    if mode == "linear":
        return _linear(la_patch, xc, xq)
    return _pchip(la_patch, xc, xq)


def _linear(y: torch.Tensor, xc: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Batched ``np.interp`` on increasing ``xc``."""
    # rightmost interval index i with xc[i] <= xq (clipped already ⇒ in range)
    i = torch.searchsorted(xc, xq, right=True).clamp(1, xc.numel() - 1) - 1
    x0, x1 = xc[i], xc[i + 1]
    w = (xq - x0) / (x1 - x0).clamp_min(torch.finfo(y.dtype).eps)
    y0, y1 = y[:, i], y[:, i + 1]
    return y0 * (1 - w) + y1 * w


def _pchip_derivatives(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """SciPy-compatible PCHIP derivatives at knots; ``y`` is ``(B, n)``."""
    B, n = y.shape
    h = x[1:] - x[:-1]                          # (n-1,)
    m = (y[:, 1:] - y[:, :-1]) / h              # (B, n-1)
    d = torch.zeros_like(y)

    if n == 2:
        d[:, 0] = m[:, 0]
        d[:, 1] = m[:, 0]
        return d

    # Interior (Fritsch–Carlson weighted harmonic mean).
    mk0, mk1 = m[:, :-1], m[:, 1:]
    hk0, hk1 = h[:-1], h[1:]
    prod = mk0 * mk1
    w1 = 2 * hk1 + hk0
    w2 = hk1 + 2 * hk0
    harm = (w1 + w2) / (w1 / mk0 + w2 / mk1)
    d[:, 1:-1] = torch.where(prod > 0, harm, torch.zeros_like(harm))

    # Endpoints (SciPy _edge_case).
    d[:, 0] = _edge_case(h[0], h[1], m[:, 0], m[:, 1])
    d[:, -1] = _edge_case(h[-1], h[-2], m[:, -1], m[:, -2])
    return d


def _edge_case(
    h0: torch.Tensor,
    h1: torch.Tensor,
    m0: torch.Tensor,
    m1: torch.Tensor,
) -> torch.Tensor:
    d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
    d = torch.where(torch.sign(d) != torch.sign(m0), torch.zeros_like(d), d)
    too_steep = (torch.sign(m0) != torch.sign(m1)) & (d.abs() > 3 * m0.abs())
    d = torch.where(too_steep, 3 * m0, d)
    return d


def _pchip(y: torch.Tensor, xc: torch.Tensor, xq: torch.Tensor) -> torch.Tensor:
    """Evaluate PCHIP of ``y (B, n)`` at ``xq``; knots ``xc`` increasing."""
    d = _pchip_derivatives(xc, y)
    i = torch.searchsorted(xc, xq, right=True).clamp(1, xc.numel() - 1) - 1
    x0, x1 = xc[i], xc[i + 1]
    h = x1 - x0
    t = (xq - x0) / h.clamp_min(torch.finfo(y.dtype).eps)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2
    y0, y1 = y[:, i], y[:, i + 1]
    d0, d1 = d[:, i], d[:, i + 1]
    return h00 * y0 + h10 * h * d0 + h01 * y1 + h11 * h * d1
