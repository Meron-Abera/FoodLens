"""Cuisine fairness audit: per-group accuracy, calibration and error analysis."""
from .metrics import (
    bootstrap_group_accuracy,
    calibration_by_group,
    compare_models,
    error_breakdown,
    expected_calibration_error,
    group_confusion_matrix,
    group_performance,
    load_audit_frame,
    load_cuisine_map,
    most_confused_pairs,
    paired_gap_change,
    per_dish_metrics,
    representation_table,
    save_cuisine_reliability,
    validate_audit_frame,
)
from .plots import (
    plot_calibration,
    plot_group_accuracy,
    plot_group_confusion,
    plot_per_dish_spread,
    plot_representation,
)
