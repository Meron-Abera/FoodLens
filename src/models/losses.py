"""Loss functions."""
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017): cross-entropy scaled by (1 - p)^gamma.

    p is the probability the model gives the correct class, so images the model already
    classifies confidently contribute almost nothing and training focuses on hard images.
    With gamma = 0 it is ordinary cross-entropy.

    Args:
        gamma: How strongly easy examples are down-weighted. 2.0 is the value from the paper.
    """

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        """Computes the mean focal loss over a batch.

        Args:
            logits: Raw model outputs, shape (N, C).
            targets: Correct class index for each example, shape (N,).

        Returns:
            The mean loss as a scalar tensor.
        """
        log_p = F.log_softmax(logits, dim=1).gather(1, targets.unsqueeze(1)).squeeze(1)
        p = log_p.exp()
        return (-(1 - p) ** self.gamma * log_p).mean()
