"""Tests for src.audit using a tiny made-up prediction table."""
import numpy as np
import pandas as pd
import pytest

from src.audit import (
    calibration_by_group,
    error_breakdown,
    expected_calibration_error,
    group_confusion_matrix,
    group_performance,
    most_confused_pairs,
    paired_gap_change,
    per_dish_metrics,
    representation_table,
    validate_audit_frame,
)


@pytest.fixture
def cuisine():
    return pd.DataFrame({'label': ['pizza', 'burger', 'ramen', 'pho'],
                         'cuisine_group': ['Western', 'Western', 'East Asian', 'East Asian']})


@pytest.fixture
def frame(cuisine):
    """Four images per dish. Western dishes: 4/4 right. East Asian: 2/4 right."""
    rows = []
    for label in ['pizza', 'burger']:
        rows += [(label, label, 0.9)] * 4
    for label, wrong in [('ramen', 'pho'), ('pho', 'pizza')]:
        rows += [(label, label, 0.8)] * 2 + [(label, wrong, 0.7)] * 2
    df = pd.DataFrame(rows, columns=['label', 'pred_label', 'confidence'])
    df['id'] = [f'{l}/{i}' for i, l in enumerate(df['label'])]
    df['correct'] = df['label'] == df['pred_label']
    df['top5_correct'] = True
    groups = cuisine.set_index('label')['cuisine_group']
    df['cuisine_group'] = df['label'].map(groups)
    df['pred_group'] = df['pred_label'].map(groups)
    return df


def test_validate_passes_on_consistent_data(frame):
    checks = validate_audit_frame(frame, n_dishes=4, images_per_dish=4, expected_accuracy=0.75)
    assert all(checks.values())


def test_validate_catches_wrong_row_count(frame):
    checks = validate_audit_frame(frame.iloc[:-1], n_dishes=4, images_per_dish=4)
    assert not checks['images_per_dish'] and not checks['n_rows']


def test_representation_counts_dishes(cuisine):
    table = representation_table(cuisine, images_per_dish=4)
    assert table.loc['Western', 'n_dishes'] == 2
    assert table['share_of_classes'].sum() == pytest.approx(1.0)
    assert table.loc['East Asian', 'n_test_images'] == 8


def test_precision_counts_errors_from_other_groups(frame, cuisine):
    per_dish = per_dish_metrics(frame, cuisine).set_index('label')
    # pizza is predicted 6 times, 4 correctly: the 2 wrong ones come from pho.
    assert per_dish.loc['pizza', 'precision'] == pytest.approx(4 / 6)
    assert per_dish.loc['pizza', 'recall'] == pytest.approx(1.0)


def test_group_performance_accuracy_and_gap(frame, cuisine):
    table = group_performance(frame, per_dish_metrics(frame, cuisine), n_boot=200)
    assert table.loc['Western', 'accuracy'] == pytest.approx(1.0)
    assert table.loc['East Asian', 'accuracy'] == pytest.approx(0.5)
    assert table.loc['East Asian', 'gap'] == pytest.approx(0.5 - 0.75)
    assert table.loc['East Asian', 'ci_low'] <= 0.5 <= table.loc['East Asian', 'ci_high']


def test_ece_is_zero_when_confidence_matches_accuracy():
    conf = np.array([0.5, 0.5, 1.0, 1.0])
    correct = np.array([1, 0, 1, 1])
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_ece_measures_overconfidence():
    assert expected_calibration_error(np.full(4, 0.9), np.zeros(4)) == pytest.approx(0.9)


def test_calibration_sign(frame):
    table = calibration_by_group(frame)
    assert table.loc['East Asian', 'overconfidence'] == pytest.approx(0.75 - 0.5)
    assert table.loc['Western', 'overconfidence'] == pytest.approx(0.9 - 1.0)


def test_error_breakdown_within_and_majority(frame):
    table = error_breakdown(frame, majority_group='Western')
    # East Asian errors: 2 ramen->pho (within), 2 pho->pizza (to Western).
    assert table.loc['East Asian', 'n_errors'] == 4
    assert table.loc['East Asian', 'within_cuisine'] == pytest.approx(0.5)
    assert table.loc['East Asian', 'to_majority'] == pytest.approx(0.5)
    assert table.loc['East Asian', 'majority_class_share'] == pytest.approx(0.5)


def test_group_confusion_rows_sum_to_one(frame):
    matrix = group_confusion_matrix(frame)
    assert np.allclose(matrix.sum(axis=1), 1.0)
    assert matrix.loc['East Asian', 'Western'] == pytest.approx(0.25)


def test_most_confused_pairs(frame):
    pairs = most_confused_pairs(frame, per_group=1)
    assert len(pairs) == 1
    assert pairs.iloc[0]['count'] == 2


def test_paired_gap_change_needs_same_images(frame):
    with pytest.raises(ValueError):
        paired_gap_change(frame, frame.iloc[::-1].reset_index(drop=True))


def test_paired_gap_change_detects_narrower_gap(frame):
    fairer = frame.copy()
    fairer.loc[fairer['label'] == 'pho', 'correct'] = True
    result = paired_gap_change(frame, fairer, n_boot=200)
    assert result['gap_a'] == pytest.approx(0.5)
    assert result['gap_b'] == pytest.approx(0.25)
    assert result['change_ci_high'] <= 0
