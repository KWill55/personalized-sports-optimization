"""
plots.py — evaluation figure helpers

Generates confusion matrices, PR curves, ROC curves, and calibration plots.
Also dumps underlying curve points to CSV for later comparison.

TODO eventually get rid of this. This is old stuff 
"""

import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    roc_curve, RocCurveDisplay,
)
from sklearn.calibration import calibration_curve

# -------------------------------------------------------------------
# helpers
# -------------------------------------------------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _as_prob_matrix(y_proba, n_classes: int) -> np.ndarray:
    """Ensure proba is shape (N, n_classes)."""
    P = np.asarray(y_proba)
    if P.ndim == 1:  # only positive-class probs
        P = np.column_stack([1 - P, P])
    if P.shape[1] != n_classes:
        raise ValueError(f"y_proba has {P.shape[1]} cols, expected {n_classes}")
    return P

# -------------------------------------------------------------------
# confusion matrices
# -------------------------------------------------------------------
def save_confusion_matrices(y_true, y_pred, class_names, outdir: Path):
    outdir = _ensure_dir(outdir)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(4,4), dpi=150)
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png")
    fig.savefig(outdir / "confusion_matrix.svg")
    plt.close(fig)

# -------------------------------------------------------------------
# precision-recall curves
# -------------------------------------------------------------------
def save_pr_curves(y_true, y_proba, class_names, outdir: Path):
    outdir = _ensure_dir(outdir)
    n_classes = len(class_names)
    P = _as_prob_matrix(y_proba, n_classes)

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    if n_classes == 2:
        prob_pos = P[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, prob_pos)
        ap = average_precision_score(y_true, prob_pos)
        ax.plot(recall, precision, label=f"{class_names[1]} (AP={ap:.3f})")
        fig.tight_layout()

        # save curve points + AP
        np.savetxt(outdir / "pr_curve_points.csv",
                   np.column_stack([recall, precision]),
                   delimiter=",", header="recall,precision", comments="")
        np.savetxt(outdir / "average_precision_per_class.csv",
                   np.array([[1, ap]]),
                   delimiter=",", header="class_index,AP", comments="", fmt=["%d","%.6f"])
    else:
        # TODO: extend for multiclass
        pass

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    fig.savefig(outdir / "pr_curve.png")
    fig.savefig(outdir / "pr_curve.svg")
    plt.close(fig)

# -------------------------------------------------------------------
# ROC curves
# -------------------------------------------------------------------
def save_roc_curves(y_true, y_proba, class_names, outdir: Path):
    outdir = _ensure_dir(outdir)
    n_classes = len(class_names)
    P = _as_prob_matrix(y_proba, n_classes)

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, P[:, 1])
        RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)
        ax.set_title("ROC Curve")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(outdir / "roc_curve.png")
        fig.savefig(outdir / "roc_curve.svg")
        np.savetxt(outdir / "roc_curve_points.csv",
                   np.column_stack([fpr, tpr]),
                   delimiter=",", header="fpr,tpr", comments="")
        plt.close(fig)
    else:
        pass  # multiclass later

# -------------------------------------------------------------------
# calibration plots
# -------------------------------------------------------------------
def save_calibration_plots(y_true, y_proba, class_names, outdir: Path,
                           n_bins: int = 10, strategy: str = "uniform"):
    outdir = _ensure_dir(outdir)
    n_classes = len(class_names)
    P = _as_prob_matrix(y_proba, n_classes)

    if n_classes == 2:
        fig, ax = plt.subplots(figsize=(5,4), dpi=150)
        frac_pos, mean_pred = calibration_curve(y_true, P[:, 1],
                                                n_bins=n_bins, strategy=strategy)
        ax.plot(mean_pred, frac_pos, "s-", label=class_names[1])
        ax.plot([0,1], [0,1], "k--", alpha=0.7)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.legend()
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(outdir / "calibration_curve.png")
        fig.savefig(outdir / "calibration_curve.svg")

        np.savetxt(outdir / "calibration_points.csv",
                   np.column_stack([mean_pred, frac_pos]),
                   delimiter=",", header="mean_pred,frac_pos", comments="")
        plt.close(fig)

# -------------------------------------------------------------------
# one-call aggregator
# -------------------------------------------------------------------
def save_all_eval_figures(outdir: Path, y_true, y_pred, y_proba, class_names):
    outdir = _ensure_dir(Path(outdir))
    save_confusion_matrices(y_true, y_pred, class_names, outdir)
    save_pr_curves(y_true, y_proba, class_names, outdir)
    save_roc_curves(y_true, y_proba, class_names, outdir)
    save_calibration_plots(y_true, y_proba, class_names, outdir)
    return outdir
