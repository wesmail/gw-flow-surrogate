"""Tests for loss functions (C2, H3)."""

import math

import torch
from torch.distributions import StudentT

from models.heads import log_i0
from models.losses import student_t_nll


def test_student_t_nll_is_proper_nll():
  target = torch.tensor([0.0, 1.0, -0.5])
  mu = torch.tensor([0.1, 0.9, -0.4])
  log_s = torch.log(torch.tensor([0.5, 0.3, 0.4]))
  nu = torch.tensor([3.0, 4.0, 5.0])
  ours = student_t_nll(target, mu, log_s, nu)
  ref = -StudentT(nu, mu, torch.exp(log_s)).log_prob(target)
  assert torch.allclose(ours, ref, atol=1e-5)


def test_student_t_nll_decreases_as_mu_approaches_target():
  target = torch.zeros(1)
  log_s = torch.zeros(1)
  nu = torch.tensor([5.0])
  far = student_t_nll(target, torch.tensor([2.0]), log_s, nu)
  near = student_t_nll(target, torch.tensor([0.01]), log_s, nu)
  assert near.item() < far.item()


def test_log_i0_matches_bessel():
  for k in [1e-2, 0.1, 1.0, 10.0, 100.0]:
    kappa = torch.tensor([k])
    approx = log_i0(kappa)
    exact = torch.special.i0e(kappa).log() + kappa
    assert torch.allclose(approx, exact, atol=1e-4)
