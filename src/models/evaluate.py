"""Test-set predictions and overall metrics."""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


@torch.no_grad()
def predict(model, loader, device):
    """Predicts class probabilities for every image in `loader`.

    Args:
        model: Trained classifier that returns logits.
        loader: DataLoader yielding (images, targets) batches; targets are ignored.
        device: Device the model is on.

    Returns:
        Softmax probabilities as a NumPy array of shape (N, num_classes), in loader order.
    """
    model.eval()
    return torch.cat([F.softmax(model(x.to(device)), dim=1).cpu() for x, _ in loader]).numpy()


def predictions_table(test_df, probs, class_names):
    """Builds one row per test image from the model's probabilities.

    Args:
        test_df: Test split with 'id', 'label', 'name' and 'target' columns, in the same
            order as the rows of `probs`.
        probs: Softmax probabilities, shape (N, num_classes).
        class_names: Label for each class index.

    Returns:
        DataFrame with the true and predicted label, the confidence (probability of the
        prediction), the probability given to the true label, whether the top-1 and top-5
        predictions are correct, and the top-5 labels.
    """
    targets = test_df['target'].to_numpy()
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]

    table = test_df[['id', 'label', 'name', 'target']].copy()
    table['pred_label'] = [class_names[i] for i in pred]
    table['confidence'] = probs.max(1)
    table['prob_true'] = probs[np.arange(len(probs)), targets]
    table['correct'] = pred == targets
    table['top5_correct'] = (top5 == targets[:, None]).any(1)
    table['top5'] = [' '.join(class_names[i] for i in row) for row in top5]
    return table


def save_predictions(name, table, probs, out_dir):
    """Writes `predictions_<name>.csv` and `probs_<name>.npy` to `out_dir`.

    Args:
        name: Model name used in the file names, e.g. 'efficientnet_b0_ce'.
        table: Output of `predictions_table`.
        probs: Softmax probabilities, shape (N, num_classes).
        out_dir: Existing folder to write the files to.
    """
    table.to_csv(out_dir / f'predictions_{name}.csv', index=False)
    np.save(out_dir / f'probs_{name}.npy', probs)


def overall_metrics(tables):
    """Summarizes each model's predictions table.

    Args:
        tables: Dict mapping a model name to its `predictions_table`.

    Returns:
        DataFrame indexed by model with top-1 and top-5 accuracy, macro precision,
        recall and F1, and mean confidence.
    """
    rows = []
    for name, table in tables.items():
        precision, recall, f1, _ = precision_recall_fscore_support(
            table['label'], table['pred_label'], average='macro', zero_division=0)
        rows.append({'model': name,
                     'top-1 accuracy': table['correct'].mean(),
                     'top-5 accuracy': table['top5_correct'].mean(),
                     'macro precision': precision, 'macro recall': recall, 'macro F1': f1,
                     'mean confidence': table['confidence'].mean()})
    return pd.DataFrame(rows).set_index('model').round(4)
