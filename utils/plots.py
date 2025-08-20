# scripts/utils/plots.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import label_binarize

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _as_prob_matrix(y_proba, n_classes: int) -> np.ndarray:
    """Coerce y_proba into shape (N, C) of probabilities.
    Accepts (N,), (N,1), (N,C); handles binary and softmax/logits-ish inputs."""
    y_proba = np.asarray(y_proba)
    # 1D: interpret as positive-class prob for binary
    if y_proba.ndim == 1:
        if n_classes != 2:
            raise ValueError("1D y_proba only supported for binary problems.")
        p = y_proba.reshape(-1, 1)
        return np.hstack([1 - p, p])
    # 2D: try to fix common cases
    if y_proba.ndim == 2:
        N, C = y_proba.shape
        if C == n_classes:
            return y_proba
        if C == 1 and n_classes == 2:
            p = y_proba
            return np.hstack([1 - p, p])
        # Heuristic: treat as logits if values outside [0,1] and normalize
        if (y_proba.min() < 0) or (y_proba.max() > 1):
            z = y_proba - y_proba.max(axis=1, keepdims=True)
            e = np.exp(z)
            y_soft = e / e.sum(axis=1, keepdims=True)
            if y_soft.shape[1] == n_classes:
                return y_soft
    return y_proba  # best effort

def save_confusion_matrices(y_true, y_pred, class_names, outdir: Path):
    outdir = _ensure_dir(outdir)
    labels = list(range(len(class_names)))
    cm_counts = confusion_matrix(y_true, y_pred, labels=labels)
    for norm in [None, "true", "pred"]:
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=norm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(6,5), dpi=150)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"Confusion Matrix ({'counts' if norm is None else norm}-normalized)")
        fig.tight_layout()
        fig.savefig(outdir / f"confusion_matrix_{'counts' if norm is None else norm}.png")
        fig.savefig(outdir / f"confusion_matrix_{'counts' if norm is None else norm}.svg")
        plt.close(fig)
    np.savetxt(outdir / "confusion_matrix_counts.csv", cm_counts, delimiter=",", fmt="%d")

def save_pr_curves(y_true, y_proba, class_names, outdir: Path):
    outdir = _ensure_dir(outdir)
    n_classes = len(class_names)
    P = _as_prob_matrix(y_proba, n_classes)

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    if n_classes == 2:
        # Binary: single curve for positive class
        prob_pos = P[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, prob_pos)
        ap = average_precision_score(y_true, prob_pos)
        ax.plot(recall, precision, label=f"{class_names[1]} (AP={ap:.3f})")
        np.savetxt(outdir / "average_precision_per_class.csv",
                   np.array([[1, ap]]),
                   delimiter=",", header="class_index,AP", comments="", fmt=["%d","%.6f"])
    else:
        # Multiclass OvR
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        ap_scores = []
        for c in range(n_classes):
            precision, recall, _ = precision_recall_curve(y_bin[:, c], P[:, c])
            ap = average_precision_score(y_bin[:, c], P[:, c])
            ap_scores.append(ap)
            ax.plot(recall, precision, label=f"{class_names[c]} (AP={ap:.3f})")
        np.savetxt(outdir / "average_precision_per_class.csv",
                   np.column_stack([np.arange(n_classes), ap_scores]),
                   delimiter=",", header="class_index,AP", comments="", fmt=["%d","%.6f"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(outdir / "pr_curves.png")
    fig.savefig(outdir / "pr_curves.svg")
    plt.close(fig)

def save_calibration_plots(y_true, y_proba, class_names, outdir: Path, n_bins: int = 10):
    outdir = _ensure_dir(outdir)
    n_classes = len(class_names)
    P = _as_prob_matrix(y_proba, n_classes)

    if n_classes == 2:
        fig, ax = plt.subplots(figsize=(5,4), dpi=150)
        frac_pos, mean_pred = calibration_curve(y_true, P[:, 1], n_bins=n_bins, strategy="uniform")
        ax.plot([0,1], [0,1], linestyle="--", linewidth=1)
        ax.plot(mean_pred, frac_pos, marker="o")
        ax.set_title(f"Calibration: {class_names[1]}")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Empirical fraction positive")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        fig.savefig(outdir / "calibration_reliability.png")
        fig.savefig(outdir / "calibration_reliability.svg")
        plt.close(fig)
    else:
        import math
        cols = int(math.ceil(math.sqrt(n_classes)))
        rows = int(math.ceil(n_classes / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), dpi=150)
        if not isinstance(axes, np.ndarray): axes = np.array([axes])
        axes = axes.flatten()
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for c in range(n_classes):
            frac_pos, mean_pred = calibration_curve(y_bin[:, c], P[:, c], n_bins=n_bins, strategy="uniform")
            ax = axes[c]
            ax.plot([0,1], [0,1], linestyle="--", linewidth=1)
            ax.plot(mean_pred, frac_pos, marker="o")
            ax.set_title(f"Calibration: {class_names[c]}")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Empirical fraction positive")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        for k in range(n_classes, len(axes)): axes[k].axis("off")
        fig.tight_layout()
        fig.savefig(outdir / "calibration_reliability.png")
        fig.savefig(outdir / "calibration_reliability.svg")
        plt.close(fig)

def save_all_eval_figures(outdir: Path, y_true, y_pred, y_proba, class_names):
    """Write CM (counts/normalized), PR curve(s), and calibration plot(s) to `outdir`."""
    outdir = _ensure_dir(Path(outdir))
    save_confusion_matrices(y_true, y_pred, class_names, outdir)
    save_pr_curves(y_true, y_proba, class_names, outdir)
    save_calibration_plots(y_true, y_proba, class_names, outdir)
    return outdir
