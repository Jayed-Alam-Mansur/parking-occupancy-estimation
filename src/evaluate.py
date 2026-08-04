# evaluate.py - Performance Evaluation
"""
Compute performance metrics for the occupancy estimation system.

Functions:
    compute_confusion_matrix()    - TP, FP, TN, FN counts
    compute_metrics()             - Accuracy, Precision, Recall, F1
    plot_confusion_matrix()       - Heatmap visualization
    measure_processing_time()     - Benchmark per-stage timing
    compute_fps()                 - Frames per second
    evaluate_by_weather()         - Metrics broken down by weather
    method_comparison()           - Compare different thresholding methods

Theory:
    Given ground truth labels and predicted labels:
        TP = True Positives  (correctly predicted OCCUPIED)
        FP = False Positives (predicted OCCUPIED, actually VACANT)
        TN = True Negatives  (correctly predicted VACANT)
        FN = False Negatives (predicted VACANT, actually OCCUPIED)

    Accuracy  = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)     — "of those we said occupied, how many were?"
    Recall    = TP / (TP + FN)     — "of those actually occupied, how many did we find?"
    F1        = 2·P·R / (P + R)   — harmonic mean balances both

    WHY NOT JUST ACCURACY?
    A full lot scores 90% from a trivial always-occupied predictor.
    Class-aware metrics (P, R, F1) expose this failure.
    Always report class balance alongside metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


def compute_confusion_matrix(y_true, y_pred):
    """
    Compute confusion matrix without sklearn.

    Parameters
    ----------
    y_true : list or np.ndarray
        Ground truth labels (0 or 1).
    y_pred : list or np.ndarray
        Predicted labels (0 or 1).

    Returns
    -------
    cm : dict
        Dictionary with TP, FP, TN, FN counts.
    matrix : np.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]].
    """
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    cm = {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}
    matrix = np.array([[tn, fp], [fn, tp]])

    return cm, matrix


def compute_metrics(y_true, y_pred):
    """
    Compute all performance metrics.

    Parameters
    ----------
    y_true : list or np.ndarray
        Ground truth labels.
    y_pred : list or np.ndarray
        Predicted labels.

    Returns
    -------
    metrics : dict
        Dictionary with accuracy, precision, recall, f1_score,
        and confusion matrix counts.
    """
    cm, matrix = compute_confusion_matrix(y_true, y_pred)

    total = cm['TP'] + cm['FP'] + cm['TN'] + cm['FN']
    accuracy = (cm['TP'] + cm['TN']) / total if total > 0 else 0.0

    precision = cm['TP'] / (cm['TP'] + cm['FP']) \
        if (cm['TP'] + cm['FP']) > 0 else 0.0

    recall = cm['TP'] / (cm['TP'] + cm['FN']) \
        if (cm['TP'] + cm['FN']) > 0 else 0.0

    f1 = 2 * precision * recall / (precision + recall) \
        if (precision + recall) > 0 else 0.0

    metrics = {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm,
        'matrix': matrix,
        'total_samples': total,
        'class_balance': {
            'occupied': int(np.sum(np.array(y_true) == 1)),
            'vacant': int(np.sum(np.array(y_true) == 0)),
        }
    }

    return metrics


def plot_confusion_matrix(cm, matrix=None, save_path=None, figsize=(8, 6)):
    """
    Plot confusion matrix as a heatmap.

    Parameters
    ----------
    cm : dict
        Confusion matrix counts (TP, FP, TN, FN).
    matrix : np.ndarray or None
        2x2 confusion matrix. If None, constructed from cm.
    save_path : str or None
        Path to save the figure.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if matrix is None:
        matrix = np.array([[cm['TN'], cm['FP']], [cm['FN'], cm['TP']]])

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted\nVACANT', 'Predicted\nOCCUPIED'],
                yticklabels=['Actual\nVACANT', 'Actual\nOCCUPIED'],
                ax=ax, cbar_kws={'label': 'Count'},
                annot_kws={'size': 16})

    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

    # Add TP/FP/TN/FN labels
    annotations = [
        (0.25, 0.75, f'TN = {cm["TN"]}', '#2ecc71'),
        (0.75, 0.75, f'FP = {cm["FP"]}', '#e74c3c'),
        (0.25, 0.25, f'FN = {cm["FN"]}', '#e67e22'),
        (0.75, 0.25, f'TP = {cm["TP"]}', '#3498db'),
    ]

    for x, y, text, color in annotations:
        ax.text(x, y + 0.15, text,
                transform=ax.transAxes, ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def format_metrics_report(metrics):
    """
    Format metrics as a human-readable report string.

    Parameters
    ----------
    metrics : dict
        Output from compute_metrics().

    Returns
    -------
    report : str
        Formatted text report.
    """
    lines = [
        "=" * 50,
        "  EVALUATION METRICS",
        "=" * 50,
        "",
        f"  Accuracy:   {metrics['accuracy']:.4f}  "
        f"({metrics['accuracy']*100:.1f}%)",
        f"  Precision:  {metrics['precision']:.4f}  "
        f"({metrics['precision']*100:.1f}%)",
        f"  Recall:     {metrics['recall']:.4f}  "
        f"({metrics['recall']*100:.1f}%)",
        f"  F1 Score:   {metrics['f1_score']:.4f}  "
        f"({metrics['f1_score']*100:.1f}%)",
        "",
        f"  Total Samples:  {metrics['total_samples']}",
        f"  Class Balance:  "
        f"Occupied={metrics['class_balance']['occupied']}, "
        f"Vacant={metrics['class_balance']['vacant']}",
        "",
        "  Confusion Matrix:",
        f"    TP={metrics['confusion_matrix']['TP']}  "
        f"FP={metrics['confusion_matrix']['FP']}",
        f"    FN={metrics['confusion_matrix']['FN']}  "
        f"TN={metrics['confusion_matrix']['TN']}",
        "",
        "=" * 50,
    ]

    return "\n".join(lines)


class Timer:
    """
    Context manager for benchmarking processing stages.

    Usage:
        timer = Timer()
        with timer.measure('stage_name'):
            # ... processing ...
        print(timer.report())
    """

    def __init__(self):
        self.stages = {}
        self._current_stage = None
        self._start_time = None

    class _MeasureContext:
        def __init__(self, timer, stage_name):
            self.timer = timer
            self.stage_name = stage_name

        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            elapsed = time.perf_counter() - self.start
            if self.stage_name not in self.timer.stages:
                self.timer.stages[self.stage_name] = []
            self.timer.stages[self.stage_name].append(elapsed)

    def measure(self, stage_name):
        return self._MeasureContext(self, stage_name)

    def get_mean_times(self):
        """Get mean time per stage in milliseconds."""
        return {
            name: np.mean(times) * 1000
            for name, times in self.stages.items()
        }

    def compute_fps(self):
        """Compute frames per second from total pipeline time."""
        total_ms = sum(self.get_mean_times().values())
        return 1000.0 / total_ms if total_ms > 0 else 0.0

    def report(self):
        """Format timing report."""
        mean_times = self.get_mean_times()
        total = sum(mean_times.values())

        lines = [
            "=" * 50,
            "  TIMING BREAKDOWN",
            "=" * 50,
            "",
            f"  {'Stage':<25} {'Time (ms)':<12} {'%':<8}",
            "  " + "-" * 45,
        ]

        for stage, ms in mean_times.items():
            pct = ms / total * 100 if total > 0 else 0
            lines.append(f"  {stage:<25} {ms:<12.2f} {pct:<8.1f}")

        lines.extend([
            "  " + "-" * 45,
            f"  {'TOTAL':<25} {total:<12.2f}",
            f"  {'FPS':<25} {self.compute_fps():<12.1f}",
            "",
            "=" * 50,
        ])

        return "\n".join(lines)

    def plot_breakdown(self, save_path=None, figsize=(10, 6)):
        """Create stacked bar chart of per-stage timing."""
        mean_times = self.get_mean_times()

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        stages = list(mean_times.keys())
        times = list(mean_times.values())

        colors = plt.cm.Set3(np.linspace(0, 1, len(stages)))
        bars = ax.barh(stages, times, color=colors, edgecolor='gray')

        for bar, t in zip(bars, times):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{t:.1f} ms', va='center', fontsize=9)

        ax.set_xlabel('Time (ms)')
        ax.set_title(f'Per-Stage Timing (FPS = {self.compute_fps():.1f})',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
