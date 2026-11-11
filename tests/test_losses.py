"""Tests for src/models/losses.py."""
import pytest
import torch
import torch.nn.functional as F

from src.models.losses import FocalLoss

# Softmax of these logits gives the correct class (0) a probability of ~0.9 and ~0.1.
EASY = torch.tensor([[2.2, 0.0]])
HARD = torch.tensor([[-2.2, 0.0]])
TARGET = torch.tensor([0])


def test_gamma_zero_equals_cross_entropy():
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    assert FocalLoss(gamma=0)(logits, targets) == pytest.approx(F.cross_entropy(logits, targets).item())


def test_matches_formula():
    p = F.softmax(HARD, dim=1)[0, 0].item()
    expected = -(1 - p) ** 2 * torch.log(torch.tensor(p)).item()
    assert FocalLoss(gamma=2)(HARD, TARGET).item() == pytest.approx(expected, rel=1e-5)


def test_down_weights_easy_examples_far_more_than_hard_ones():
    focal, ce = FocalLoss(gamma=2), F.cross_entropy
    easy_ratio = focal(EASY, TARGET) / ce(EASY, TARGET)
    hard_ratio = focal(HARD, TARGET) / ce(HARD, TARGET)
    assert easy_ratio < 0.02   # (1 - 0.9)^2 = 0.01
    assert hard_ratio > 0.8    # (1 - 0.1)^2 = 0.81


def test_never_exceeds_cross_entropy():
    logits = torch.randn(32, 10)
    targets = torch.randint(0, 10, (32,))
    per_example_focal = torch.stack([FocalLoss()(logits[i:i + 1], targets[i:i + 1]) for i in range(32)])
    assert torch.all(per_example_focal <= F.cross_entropy(logits, targets, reduction='none') + 1e-6)


def test_has_gradients():
    logits = torch.randn(4, 3, requires_grad=True)
    FocalLoss()(logits, torch.tensor([0, 1, 2, 0])).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
