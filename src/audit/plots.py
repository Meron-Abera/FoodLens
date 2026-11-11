"""Figures for the cuisine audit. Each function draws one chart and returns its figure."""
import matplotlib.pyplot as plt
import numpy as np

from .metrics import GROUP_COL

MODEL_COLORS = ['#2F5D8F', '#D9582B']


def plot_representation(representation):
    """Bar chart of how many of the 101 dishes belong to each cuisine group.

    Args:
        representation: Output of ``representation_table``.

    Returns:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    table = representation.sort_values('n_dishes')
    bars = ax.barh(table.index, table['n_dishes'], color='#8C8580')
    for bar, share in zip(bars, table['share_of_classes']):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{int(bar.get_width())} ({share:.0%})', va='center', fontsize=9)
    ax.set_xlabel('Number of dishes (each has 250 test images)')
    ax.set_title('Dishes per cuisine group in Food-101')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    return fig


def plot_group_accuracy(tables, overall):
    """Accuracy per cuisine group with 95% intervals, models side by side.

    Args:
        tables: Dict mapping model name to its ``group_performance`` table.
        overall: Dict mapping model name to its overall accuracy.

    Returns:
        The matplotlib figure.
    """
    first = next(iter(tables.values()))
    order = first.sort_values('accuracy').index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, ((name, table), color) in enumerate(zip(tables.items(), MODEL_COLORS)):
        table = table.loc[order]
        err = [table['accuracy'] - table['ci_low'], table['ci_high'] - table['accuracy']]
        shift = (k - (len(tables) - 1) / 2) * 0.2
        ax.errorbar(table['accuracy'], y + shift, xerr=err, fmt='o', capsize=3, color=color, label=name)
        ax.axvline(overall[name], color=color, ls=':', lw=1)
    ax.set_yticks(y, order)
    ax.set_xlabel('Top-1 accuracy (95% bootstrap interval); dotted line = overall')
    ax.set_title('Accuracy by cuisine group')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_per_dish_spread(per_dish):
    """Strip plot of per-dish accuracy within each cuisine group.

    Shows whether a low group average comes from the whole cuisine or from a
    few hard dishes.

    Args:
        per_dish: Output of ``per_dish_metrics``.

    Returns:
        The matplotlib figure.
    """
    order = per_dish.groupby(GROUP_COL)['accuracy'].median().sort_values().index
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rng = np.random.default_rng(0)
    for i, group in enumerate(order):
        part = per_dish[per_dish[GROUP_COL] == group]
        ax.scatter(part['accuracy'], i + rng.uniform(-0.15, 0.15, len(part)), s=18, alpha=0.7, color='#2F5D8F')
        ax.plot([part['accuracy'].median()] * 2, [i - 0.3, i + 0.3], color='#D9582B', lw=2)
        worst = part.nsmallest(1, 'accuracy').iloc[0]
        ax.annotate(worst['label'], (worst['accuracy'], i), xytext=(-4, 8),
                    textcoords='offset points', ha='right', fontsize=7, color='#6B6560')
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel('Per-dish accuracy (orange = group median, label = hardest dish)')
    ax.set_title('Accuracy of each dish within its cuisine group')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_calibration(calibrations):
    """Overconfidence per cuisine group for each model.

    Args:
        calibrations: Dict mapping model name to its ``calibration_by_group`` table.

    Returns:
        The matplotlib figure.
    """
    first = next(iter(calibrations.values()))
    order = first.sort_values('overconfidence').index
    y = np.arange(len(order))
    height = 0.8 / len(calibrations)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, ((name, table), color) in enumerate(zip(calibrations.items(), MODEL_COLORS)):
        ax.barh(y + (k - (len(calibrations) - 1) / 2) * height, table.loc[order, 'overconfidence'],
                height=height, color=color, label=name)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y, order)
    ax.set_xlabel('Mean confidence minus accuracy (positive = overconfident)')
    ax.set_title('Calibration by cuisine group')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


def plot_group_confusion(matrix, title='Where predictions go, by cuisine group'):
    """Heatmap of the row-normalized cuisine-group confusion matrix.

    Args:
        matrix: Output of ``group_confusion_matrix`` with ``normalize=True``.
        title: Chart title.

    Returns:
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix.to_numpy(), cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iat[i, j]
            ax.text(j, i, f'{value:.1%}' if value < 0.995 else '>99%', ha='center', va='center',
                    fontsize=8, color='white' if value > 0.6 else 'black')
    ax.set_xlabel('Predicted dish\'s cuisine')
    ax.set_ylabel('True cuisine')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig
