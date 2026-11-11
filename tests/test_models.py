"""Tests for src/models/architectures.py and src/models/train.py.

Models are built without pretrained weights and trained on tiny random tensors,
so these tests run quickly on a CPU without downloading anything.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.architectures import build_efficientnet_b0, load_trained_efficientnet_b0
from src.models.losses import FocalLoss
from src.models.train import fine_tune, run_epoch, train_or_load

CONFIG = {'epochs': 2, 'lr_backbone': 1e-4, 'lr_head': 1e-3, 'weight_decay': 1e-4}


def tiny_loader(num_classes=3, n=6):
    x = torch.randn(n, 3, 64, 64)
    y = torch.arange(n) % num_classes
    return DataLoader(TensorDataset(x, y), batch_size=3)


def test_build_replaces_head_for_our_classes():
    model, head = build_efficientnet_b0(101, pretrained=False)
    assert head is model.classifier
    assert head[1].out_features == 101
    assert model.eval()(torch.randn(2, 3, 224, 224)).shape == (2, 101)


def test_run_epoch_without_optimizer_does_not_change_weights():
    model, _ = build_efficientnet_b0(3, pretrained=False)
    before = [p.clone() for p in model.parameters()]
    loss, acc = run_epoch(model, tiny_loader(), nn.CrossEntropyLoss(), 'cpu')
    assert loss > 0 and 0 <= acc <= 1
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_fine_tune_saves_checkpoint_that_reloads(tmp_path):
    model, head = build_efficientnet_b0(3, pretrained=False)
    checkpoint = tmp_path / 'tiny.pt'
    history = fine_tune(model, head, FocalLoss(), tiny_loader(), tiny_loader(), checkpoint, CONFIG, 'cpu')

    assert [h['epoch'] for h in history] == [1, 2]
    reloaded, num_classes, saved_history = load_trained_efficientnet_b0(checkpoint)
    assert num_classes == 3 and len(saved_history) == 2
    x = torch.randn(2, 3, 64, 64)
    assert torch.allclose(model.eval()(x), reloaded(x), atol=1e-5)


def test_train_or_load_loads_existing_checkpoint_instead_of_training(tmp_path):
    model, head = build_efficientnet_b0(3, pretrained=False)
    fine_tune(model, head, nn.CrossEntropyLoss(), tiny_loader(), tiny_loader(), tmp_path / 'tiny.pt', CONFIG, 'cpu')

    # Loaders that fail if used prove the model is loaded, not retrained.
    loaded, history = train_or_load('tiny', nn.CrossEntropyLoss(), None, None, 3, CONFIG, 'cpu', tmp_path)
    assert len(history) == 2
    x = torch.randn(1, 3, 64, 64)
    assert torch.allclose(model.eval()(x), loaded(x), atol=1e-5)
