"""Tests for src/models/export_coreml.py that don't need coremltools."""
import torch

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from src.models.architectures import build_efficientnet_b0
from src.models.export_coreml import DeployableClassifier, class_labels


def test_class_labels_match_training_order(tiny_dataset):
    _, index_csv = tiny_dataset
    assert class_labels(index_csv, 3) == ['apple_pie', 'pizza', 'ramen']
    assert class_labels(index_csv, 2) == ['apple_pie', 'pizza']  # quick-test models


def test_deployable_classifier_normalizes_and_returns_probabilities():
    model, _ = build_efficientnet_b0(5, pretrained=False)
    model.eval()
    image = torch.rand(2, 3, 224, 224)  # pixel values in [0, 1], as Core ML passes them

    probs = DeployableClassifier(model).eval()(image)

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    expected = torch.softmax(model((image - mean) / std), dim=1)
    assert torch.allclose(probs, expected, atol=1e-6)
    assert torch.allclose(probs.sum(1), torch.ones(2))
