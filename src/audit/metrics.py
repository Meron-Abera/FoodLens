"""Cuisine-level fairness audit for FoodLens test-set predictions.

Every function takes the per-image prediction table written by training
(one row per test image) joined with the cuisine mapping, so the audit can be
rerun for any model that writes the same prediction CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

GROUP_COL = 'cuisine_group'
PRED_GROUP_COL = 'pred_group'


def _as_bool(values):
    """Convert a column of True/False values read from CSV to booleans."""
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin(['true', '1'])


def load_cuisine_map(path):
    """Read the cuisine mapping and check it covers each dish exactly once.

    Args:
        path: CSV with at least the columns ``label`` and ``cuisine_group``.

    Returns:
        The mapping as a DataFrame.

    Raises:
        ValueError: If a label appears more than once or a group is missing.
    """
    cuisine = pd.read_csv(path)
    if cuisine['label'].duplicated().any():
        dupes = cuisine.loc[cuisine['label'].duplicated(), 'label'].tolist()
        raise ValueError(f'Labels mapped more than once: {dupes}')
    if cuisine[GROUP_COL].isna().any():
        raise ValueError('Every label needs a cuisine_group.')
    return cuisine


def load_audit_frame(predictions_path, cuisine):
    """Join a prediction CSV with the cuisine of the true and predicted dish.

    Args:
        predictions_path: Prediction CSV written by training.
        cuisine: Mapping returned by :func:`load_cuisine_map`.

    Returns:
        The predictions with ``cuisine_group`` (true dish) and ``pred_group``
        (predicted dish) columns added. ``correct`` and ``top5_correct`` are booleans.

    Raises:
        ValueError: If a true or predicted label has no cuisine group.
    """
    preds = pd.read_csv(predictions_path)
    for col in ('correct', 'top5_correct'):
        preds[col] = _as_bool(preds[col])
    groups = cuisine[['label', GROUP_COL]]
    df = preds.merge(groups, on='label', how='left')
    df = df.merge(groups.rename(columns={'label': 'pred_label', GROUP_COL: PRED_GROUP_COL}),
                  on='pred_label', how='left')
    missing = sorted(set(df.loc[df[GROUP_COL].isna(), 'label']) | set(df.loc[df[PRED_GROUP_COL].isna(), 'pred_label']))
    if missing:
        raise ValueError(f'Labels missing from the cuisine map: {missing}')
    return df


def validate_audit_frame(df, n_dishes=101, images_per_dish=250, expected_accuracy=None, tol=1e-3):
    """Run the sanity checks from step 1 of the audit guide.

    Args:
        df: Output of :func:`load_audit_frame`.
        n_dishes: Expected number of distinct dishes.
        images_per_dish: Expected test images per dish.
        expected_accuracy: Overall top-1 accuracy reported by training, if known.
        tol: Allowed difference from ``expected_accuracy``.

    Returns:
        A dict of check name to result, all ``True`` when the data is consistent.
    """
    counts = df['label'].value_counts()
    checks = {
        'all_labels_mapped': bool(df[GROUP_COL].notna().all()),
        'n_dishes': counts.size == n_dishes,
        'images_per_dish': bool((counts == images_per_dish).all()),
        'n_rows': len(df) == n_dishes * images_per_dish,
    }
    if expected_accuracy is not None:
        checks['accuracy_matches_training'] = abs(df['correct'].mean() - expected_accuracy) <= tol
    return checks


def representation_table(cuisine, images_per_dish=250):
    """Count dishes per cuisine group and each group's share of the classes.

    Args:
        cuisine: The cuisine mapping.
        images_per_dish: Test images per dish, used for the image count.

    Returns:
        DataFrame indexed by cuisine group, largest group first.
    """
    table = cuisine.groupby(GROUP_COL).size().rename('n_dishes').to_frame()
    table['share_of_classes'] = table['n_dishes'] / table['n_dishes'].sum()
    table['n_test_images'] = table['n_dishes'] * images_per_dish
    return table.sort_values('n_dishes', ascending=False)


def per_dish_metrics(df, cuisine):
    """Precision, recall, F1 and top-1/top-5 accuracy for every dish.

    Precision is computed over the whole test set, because a dish's precision
    depends on predictions coming from every cuisine.

    Args:
        df: Output of :func:`load_audit_frame`.
        cuisine: The cuisine mapping.

    Returns:
        One row per dish with its metrics and cuisine group.
    """
    labels = sorted(df['label'].unique())
    p, r, f1, _ = precision_recall_fscore_support(df['label'], df['pred_label'], labels=labels, zero_division=0)
    acc = df.groupby('label')[['correct', 'top5_correct']].mean()
    per_dish = pd.DataFrame({'label': labels, 'precision': p, 'recall': r, 'f1': f1})
    per_dish = per_dish.merge(acc, left_on='label', right_index=True)
    return per_dish.merge(cuisine[['label', GROUP_COL]], on='label').rename(
        columns={'correct': 'accuracy', 'top5_correct': 'top5_accuracy'})


def bootstrap_group_accuracy(df, n_boot=1000, seed=42):
    """Resample each group's images to get accuracy confidence intervals.

    Args:
        df: Output of :func:`load_audit_frame`.
        n_boot: Number of bootstrap resamples per group.
        seed: Random seed, so results are reproducible.

    Returns:
        Dict mapping each cuisine group to an array of ``n_boot`` resampled accuracies.
    """
    rng = np.random.default_rng(seed)
    samples = {}
    correct_all = df['correct'].to_numpy()
    for group, idx in df.groupby(GROUP_COL).indices.items():
        correct = correct_all[idx]
        draws = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        samples[group] = correct[draws].mean(axis=1)
    return samples


def group_performance(df, per_dish, n_boot=1000, seed=42):
    """Accuracy, precision/recall/F1 and 95% intervals for each cuisine group.

    A group is flagged when the upper end of its 95% interval is below the
    overall accuracy, i.e. it is significantly less accurate than average.

    Args:
        df: Output of :func:`load_audit_frame`.
        per_dish: Output of :func:`per_dish_metrics`.
        n_boot: Number of bootstrap resamples per group.
        seed: Random seed for the bootstrap.

    Returns:
        DataFrame indexed by cuisine group, sorted by accuracy.
    """
    overall = df['correct'].mean()
    table = df.groupby(GROUP_COL).agg(n_images=('correct', 'size'),
                                      accuracy=('correct', 'mean'),
                                      top5_accuracy=('top5_correct', 'mean'))
    table = table.join(per_dish.groupby(GROUP_COL)[['precision', 'recall', 'f1']].mean())
    samples = bootstrap_group_accuracy(df, n_boot, seed)
    table['ci_low'] = [np.percentile(samples[g], 2.5) for g in table.index]
    table['ci_high'] = [np.percentile(samples[g], 97.5) for g in table.index]
    table['gap'] = table['accuracy'] - overall
    table['flagged'] = table['ci_high'] < overall
    return table.sort_values('accuracy')


def expected_calibration_error(conf, correct, n_bins=15):
    """Weighted mean gap between confidence and accuracy over confidence bins.

    Args:
        conf: Confidence of each prediction, values in [0, 1].
        correct: Whether each prediction was right.
        n_bins: Number of equal-width confidence bins.

    Returns:
        The ECE as a float; 0 means perfectly calibrated.
    """
    conf, correct = np.asarray(conf, dtype=float), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    return float(sum(abs(correct[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                     for b in range(n_bins) if (bins == b).any()))


def calibration_by_group(df, n_bins=15):
    """Mean confidence, accuracy, overconfidence and ECE for each cuisine group.

    Args:
        df: Output of :func:`load_audit_frame`.
        n_bins: Number of confidence bins for the ECE.

    Returns:
        DataFrame indexed by cuisine group. Positive ``overconfidence`` means the
        model claims more certainty than it delivers.
    """
    rows = {}
    for group, part in df.groupby(GROUP_COL):
        rows[group] = {'mean_confidence': part['confidence'].mean(),
                       'accuracy': part['correct'].mean(),
                       'ece': expected_calibration_error(part['confidence'], part['correct'], n_bins)}
    table = pd.DataFrame(rows).T
    table['overconfidence'] = table['mean_confidence'] - table['accuracy']
    return table


def error_breakdown(df, majority_group='Western'):
    """Where each group's errors go: same cuisine, other cuisine, or the majority group.

    Args:
        df: Output of :func:`load_audit_frame`.
        majority_group: The over-represented group used in the "pulled toward the
            majority" test.

    Returns:
        DataFrame indexed by cuisine group with the number of errors, the share
        predicted as a dish from the same group, and the share predicted as a
        ``majority_group`` dish. ``majority_class_share`` is that group's share of
        all classes, the rate expected if errors were spread evenly over dishes.
    """
    wrong = df.loc[~df['correct'], [GROUP_COL, PRED_GROUP_COL]].copy()
    wrong['within_cuisine'] = wrong[PRED_GROUP_COL] == wrong[GROUP_COL]
    wrong['to_majority'] = wrong[PRED_GROUP_COL] == majority_group
    table = wrong.groupby(GROUP_COL).agg(n_errors=('within_cuisine', 'size'),
                                         within_cuisine=('within_cuisine', 'mean'),
                                         to_majority=('to_majority', 'mean'))
    classes = df.drop_duplicates('label')[GROUP_COL]
    table['majority_class_share'] = (classes == majority_group).mean()
    return table


def group_confusion_matrix(df, normalize=True):
    """Confusion matrix between cuisine groups (true group by predicted group).

    Args:
        df: Output of :func:`load_audit_frame`.
        normalize: If true, each row sums to 1.

    Returns:
        Square DataFrame with true groups as rows and predicted groups as columns.
    """
    matrix = pd.crosstab(df[GROUP_COL], df[PRED_GROUP_COL])
    order = sorted(set(matrix.index) | set(matrix.columns))
    matrix = matrix.reindex(index=order, columns=order, fill_value=0)
    return matrix.div(matrix.sum(axis=1), axis=0) if normalize else matrix


def most_confused_pairs(df, per_group=3):
    """The most frequent wrong predictions for each cuisine group.

    Args:
        df: Output of :func:`load_audit_frame`.
        per_group: Number of pairs to keep per group.

    Returns:
        DataFrame with the true dish, predicted dish, count and the share of the
        true dish's test images that ended up as that prediction.
    """
    wrong = df[~df['correct']]
    pairs = wrong.groupby([GROUP_COL, 'label', 'pred_label', PRED_GROUP_COL]).size().rename('count').reset_index()
    pairs['share_of_dish'] = pairs['count'] / pairs['label'].map(df['label'].value_counts())
    pairs = pairs.sort_values([GROUP_COL, 'count'], ascending=[True, False])
    return pairs.groupby(GROUP_COL).head(per_group).reset_index(drop=True)


def paired_gap_change(df_a, df_b, n_boot=1000, seed=42):
    """Bootstrap the change in best-minus-worst group accuracy between two models.

    Both models are scored on the same resampled images, which is valid because
    their prediction files list the same test images in the same order.

    Args:
        df_a: Audit frame for the reference model (e.g. cross-entropy).
        df_b: Audit frame for the comparison model (e.g. focal loss).
        n_boot: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with each model's gap and the change (b minus a) with its 95% interval.

    Raises:
        ValueError: If the two frames do not list the same images in the same order.
    """
    if not df_a['id'].equals(df_b['id']):
        raise ValueError('Prediction files must list the same test images in the same order.')
    rng = np.random.default_rng(seed)
    gaps_a, gaps_b = [], []
    groups = df_a.groupby(GROUP_COL).indices
    draws = {g: rng.integers(0, len(idx), size=(n_boot, len(idx))) for g, idx in groups.items()}
    for df, gaps in ((df_a, gaps_a), (df_b, gaps_b)):
        correct = df['correct'].to_numpy()
        accs = np.stack([correct[idx][draws[g]].mean(axis=1) for g, idx in groups.items()], axis=1)
        gaps.extend(accs.max(axis=1) - accs.min(axis=1))
    point = {name: d.groupby(GROUP_COL)['correct'].mean() for name, d in (('a', df_a), ('b', df_b))}
    gap_a = point['a'].max() - point['a'].min()
    gap_b = point['b'].max() - point['b'].min()
    change = np.array(gaps_b) - np.array(gaps_a)
    return {'gap_a': gap_a, 'gap_b': gap_b, 'change': gap_b - gap_a,
            'change_ci_low': np.percentile(change, 2.5), 'change_ci_high': np.percentile(change, 97.5)}


def compare_models(tables, overall):
    """Summarize the fairness of several models in one table (step 7).

    Args:
        tables: Dict mapping model name to its :func:`group_performance` table.
        overall: Dict mapping model name to its overall top-1 accuracy.

    Returns:
        DataFrame indexed by model with overall accuracy, worst-group accuracy and
        the best-minus-worst gap.
    """
    rows = {}
    for name, table in tables.items():
        rows[name] = {'overall_accuracy': overall[name],
                      'worst_group': table['accuracy'].idxmin(),
                      'worst_group_accuracy': table['accuracy'].min(),
                      'best_group': table['accuracy'].idxmax(),
                      'max_min_gap': table['accuracy'].max() - table['accuracy'].min(),
                      'n_flagged': int(table['flagged'].sum())}
    return pd.DataFrame(rows).T


def save_cuisine_reliability(performance, calibration, path):
    """Write the per-cuisine reliability file used by the app (step 8).

    Args:
        performance: Output of :func:`group_performance` for the deployed model.
        calibration: Output of :func:`calibration_by_group` for the same model.
        path: Where to write the CSV.

    Returns:
        The table that was written.
    """
    table = performance[['accuracy', 'flagged']].join(calibration[['ece']])
    table = table.rename_axis(GROUP_COL).reset_index()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table[[GROUP_COL, 'accuracy', 'ece', 'flagged']].to_csv(path, index=False)
    return table
