"""
tests/test_uncertainty.py
==========================
Tests for the uncertainty-aware neural network and Gaussian NLL loss.
Runs without a GPU — uses a tiny 1-block network for speed.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

pytestmark = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")


@pytest.fixture
def tiny_model():
    from p2_neural_network.model import ChaturangaNet
    return ChaturangaNet(
        in_channels=13, num_filters=8, num_res_blocks=1,
        action_size=4096, uncertainty_head=True
    )

@pytest.fixture
def tiny_model_no_unc():
    from p2_neural_network.model import ChaturangaNet
    return ChaturangaNet(
        in_channels=13, num_filters=8, num_res_blocks=1,
        action_size=4096, uncertainty_head=False
    )


# ── Architecture tests ────────────────────────────────────────────────────────

def test_uncertainty_forward_returns_three_tensors(tiny_model):
    import torch
    x = torch.randn(2, 13, 8, 8)
    out = tiny_model(x)
    assert len(out) == 3, "uncertainty_head=True should return 3 outputs"

def test_no_uncertainty_forward_returns_two_tensors(tiny_model_no_unc):
    import torch
    x = torch.randn(2, 13, 8, 8)
    out = tiny_model_no_unc(x)
    assert len(out) == 2, "uncertainty_head=False should return 2 outputs"

def test_policy_shape(tiny_model):
    import torch
    x = torch.randn(1, 13, 8, 8)
    policy_logits, value_mean, value_logvar = tiny_model(x)
    assert policy_logits.shape == (1, 4096)

def test_value_mean_range(tiny_model):
    import torch
    x = torch.randn(4, 13, 8, 8)
    _, value_mean, _ = tiny_model(x)
    # tanh should keep values in (-1, 1)
    assert (value_mean.abs() <= 1.0 + 1e-5).all()

def test_value_logvar_unconstrained(tiny_model):
    import torch
    x = torch.randn(4, 13, 8, 8)
    _, _, logvar = tiny_model(x)
    # logvar should be unconstrained (can be negative)
    assert logvar.shape == (4, 1)


# ── predict() API ─────────────────────────────────────────────────────────────

def test_predict_with_uncertainty_returns_three(tiny_model):
    import torch
    x = torch.randn(1, 13, 8, 8)
    out = tiny_model.predict(x)
    assert len(out) == 3, "predict() with uncertainty_head should return 3 values"

def test_predict_no_uncertainty_returns_two(tiny_model_no_unc):
    import torch
    x = torch.randn(1, 13, 8, 8)
    out = tiny_model_no_unc.predict(x)
    assert len(out) == 2

def test_predict_value_std_positive(tiny_model):
    import torch
    x = torch.randn(1, 13, 8, 8)
    _, value_mean, value_std = tiny_model.predict(x)
    assert value_std > 0, "value_std must be positive"

def test_predict_policy_sums_to_one(tiny_model):
    import torch
    x = torch.randn(1, 13, 8, 8)
    probs, _, _ = tiny_model.predict(x)
    assert abs(probs.sum().item() - 1.0) < 1e-4

def test_predict_with_mask(tiny_model):
    import torch
    x    = torch.randn(1, 13, 8, 8)
    mask = torch.zeros(4096, dtype=torch.bool)
    mask[10:20] = True   # only 10 legal actions
    probs, _, _ = tiny_model.predict(x, mask)
    # Illegal actions should be ~0
    assert probs[:10].sum().item() < 1e-5


# ── Gaussian NLL loss ─────────────────────────────────────────────────────────

def test_gaussian_nll_loss_shape():
    import torch
    from p2_neural_network.train_utils import gaussian_nll_loss
    mean   = torch.tensor([[0.5], [-0.3], [0.1]])
    logvar = torch.zeros(3, 1)
    target = torch.tensor([[1.0], [-1.0], [0.0]])
    loss = gaussian_nll_loss(mean, logvar, target)
    assert loss.ndim == 0, "Loss should be a scalar"

def test_gaussian_nll_loss_decreases_with_better_estimate():
    import torch
    from p2_neural_network.train_utils import gaussian_nll_loss
    target = torch.tensor([[1.0]])
    logvar = torch.zeros(1, 1)
    good_pred  = torch.tensor([[0.9]])
    bad_pred   = torch.tensor([[-0.5]])
    loss_good = gaussian_nll_loss(good_pred, logvar, target).item()
    loss_bad  = gaussian_nll_loss(bad_pred,  logvar, target).item()
    assert loss_good < loss_bad

def test_compute_loss_uncertainty(tiny_model):
    import torch
    from p2_neural_network.train_utils import compute_loss
    states   = torch.randn(4, 13, 8, 8)
    policies = torch.softmax(torch.randn(4, 4096), dim=1)
    values   = torch.tensor([1.0, -1.0, 0.0, 1.0])
    losses = compute_loss(tiny_model, states, policies, values)
    assert "total_loss" in losses
    assert "policy_loss" in losses
    assert "value_loss"  in losses
    assert losses["total_loss"].item() > 0
