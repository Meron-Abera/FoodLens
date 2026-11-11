"""Export a trained FoodLens checkpoint to Core ML for the iOS app.

The exported model takes a 224x224 RGB image and returns a probability for every
dish, so the app only has to hand it a photo. ImageNet normalization and softmax
are built into the model, and the class labels are embedded in it.

Usage (from the repository root):
    python -m src.models.export_coreml --checkpoint outputs/full/efficientnet_b0_focal.pt

By default the model is written to ios/FoodLens/, where the app picks it up
automatically on the next build.
"""
import argparse
from pathlib import Path

import coremltools as ct
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from src.paths import CLEAN_INDEX, IOS_MODEL

IMAGE_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class DeployableClassifier(nn.Module):
    """Wraps the trained network with the preprocessing and softmax the app needs.

    Args:
        model: Trained classifier that maps normalized images to logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, image):
        """Classifies a batch of RGB images with pixel values in [0, 1]."""
        return torch.softmax(self.model((image - self.mean) / self.std), dim=1)


def load_checkpoint(path):
    """Rebuilds EfficientNet-B0 from a checkpoint saved by notebooks/02_models.ipynb.

    Returns:
        The model in eval mode and its number of output classes.
    """
    weights = torch.load(path, map_location='cpu')['weights']
    num_classes = weights['classifier.1.weight'].shape[0]
    model = models.efficientnet_b0()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(weights)
    return model.eval(), num_classes


def class_labels(index_csv, num_classes):
    """Returns the class labels in the order the model was trained with.

    02_models.ipynb sorts the labels alphabetically and, in quick-test mode,
    keeps the first 10, so taking the first `num_classes` sorted labels
    reproduces its order for both the full and the quick-test model.
    """
    return sorted(pd.read_csv(index_csv)['label'].unique())[:num_classes]


def export(checkpoint, output, index_csv=CLEAN_INDEX):
    """Converts a checkpoint to a Core ML classifier saved at `output`."""
    model, num_classes = load_checkpoint(checkpoint)
    labels = class_labels(index_csv, num_classes)

    example = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    traced = torch.jit.trace(DeployableClassifier(model).eval(), example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name='image', shape=example.shape, scale=1 / 255.0,
                             color_layout=ct.colorlayout.RGB)],
        classifier_config=ct.ClassifierConfig(labels),
        minimum_deployment_target=ct.target.iOS17,
    )
    mlmodel.short_description = 'FoodLens: classifies a food photo into one of the Food-101 dishes.'
    mlmodel.save(str(output))
    print(f'Saved {num_classes}-class model to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True, type=Path, help='.pt file saved by 02_models.ipynb')
    parser.add_argument('--output', type=Path, default=IOS_MODEL, help='defaults to the iOS app folder')
    parser.add_argument('--index', type=Path, default=CLEAN_INDEX, help='food101_clean.csv from 01_eda.ipynb')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export(args.checkpoint, args.output, args.index)
