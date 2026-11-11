"""Tests for src/models/evaluate.py."""
import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import overall_metrics, predictions_table, save_predictions

CLASSES = ['apple_pie', 'pizza', 'ramen', 'sushi', 'tacos', 'waffles']


@pytest.fixture
def test_df():
    return pd.DataFrame({'id': ['pizza/1', 'ramen/2', 'sushi/3'],
                         'label': ['pizza', 'ramen', 'sushi'],
                         'name': ['Pizza', 'Ramen', 'Sushi'],
                         'target': [1, 2, 3]})


@pytest.fixture
def probs():
    p = np.full((3, 6), 0.02)
    p[0, 1] = 0.90   # pizza -> pizza (correct)
    p[1, 3] = 0.60   # ramen -> sushi (wrong), ramen 2nd
    p[1, 2] = 0.30
    p[2, 3] = 0.50   # sushi -> sushi (correct)
    return p / p.sum(1, keepdims=True)


def test_predictions_table(test_df, probs):
    table = predictions_table(test_df, probs, CLASSES)
    assert table['pred_label'].tolist() == ['pizza', 'sushi', 'sushi']
    assert table['correct'].tolist() == [True, False, True]
    assert table['top5_correct'].all()
    assert table['confidence'].tolist() == pytest.approx(probs.max(1))
    assert table['prob_true'].tolist() == pytest.approx([probs[0, 1], probs[1, 2], probs[2, 3]])
    assert table['top5'][1].split()[:2] == ['sushi', 'ramen']


def test_save_predictions_writes_csv_and_npy(tmp_path, test_df, probs):
    table = predictions_table(test_df, probs, CLASSES)
    save_predictions('m', table, probs, tmp_path)
    assert pd.read_csv(tmp_path / 'predictions_m.csv')['pred_label'].tolist() == table['pred_label'].tolist()
    assert np.allclose(np.load(tmp_path / 'probs_m.npy'), probs)


def test_overall_metrics(test_df, probs):
    results = overall_metrics({'m': predictions_table(test_df, probs, CLASSES)})
    assert results.loc['m', 'top-1 accuracy'] == pytest.approx(2 / 3, abs=1e-4)
    assert results.loc['m', 'top-5 accuracy'] == 1.0
    assert 0 < results.loc['m', 'macro F1'] < 1
