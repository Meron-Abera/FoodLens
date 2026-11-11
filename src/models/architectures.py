"""Model definitions."""
import torch
import torch.nn as nn
from torchvision import models


def build_efficientnet_b0(num_classes, pretrained=True):
    """Builds EfficientNet-B0 with its final layer replaced for `num_classes` classes.

    The classifier head is torchvision's default, Dropout(0.2) -> Linear(1280 -> num_classes).

    Args:
        num_classes: Number of output classes.
        pretrained: Start from ImageNet weights (downloaded and cached on first use).

    Returns:
        (model, head): the full network and its classifier head, so the head can be given
        its own learning rate.
    """
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model, model.classifier


def load_trained_efficientnet_b0(checkpoint, device='cpu'):
    """Rebuilds a trained EfficientNet-B0 from a checkpoint saved by `src.models.train`.

    Args:
        checkpoint: Path to the `.pt` file.
        device: Device to load the weights onto, e.g. 'cpu', 'mps' or 'cuda'.

    Returns:
        The model in eval mode, its number of classes, and the training history.
    """
    saved = torch.load(checkpoint, map_location=device)
    num_classes = saved['weights']['classifier.1.weight'].shape[0]
    model, _ = build_efficientnet_b0(num_classes, pretrained=False)
    model.load_state_dict(saved['weights'])
    return model.to(device).eval(), num_classes, saved['history']
