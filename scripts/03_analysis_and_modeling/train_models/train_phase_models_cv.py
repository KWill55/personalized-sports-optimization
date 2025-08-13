#!/usr/bin/env python3
"""
train_phase_models_cv.py

- 5-fold Stratified Cross-Validation (no data leakage via Pipeline)
- Selects best model by mean F1 (positive class = made = 1)
- Saves:
    * results.csv (per-model CV means)
    * feature_importances.csv (avg across tree models, when available)
    * best_phase_model.joblib (scaler+model+feature_names in a Pipeline)
    * plots/confusion_matrix.png (from out-of-fold predictions of the best model)

Usage:
  python train_phase_models_cv.py \
    --X /path/to/X.csv \
    --y /path/to/y.csv \
    --out_dir /path/to/analysis
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
import joblib
import matplotlib.pyplot as plt
import datetime as dt

def make_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, class_weight=None))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, class_weight=None))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=42))
        ]),
        "Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB())
        ]),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", required=True, help="Path to X.csv")
    ap.add_argument("--y", required=True, help="Path to y.csv (single column 0/1)")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X = pd.read_csv(args.X)
    y = pd.read_csv(args.y).squeeze()
    X = X.select_dtypes(include=[np.number]).copy()
    feature_names = X.columns.tolist()

    # CV setup
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=42)

    models = make_models()

    # Metrics to compute via cross_validate
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results_rows = []
    feat_imps_rows = []

    # Evaluate all models via CV
    for name, pipe in models.items():
        cv_res = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=None, return_estimator=True)
        results_rows.append({
            "Model": name,
            "CV_Accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
            "CV_Precision_mean": float(np.mean(cv_res["test_precision"])),
            "CV_Recall_mean": float(np.mean(cv_res["test_recall"])),
            "CV_F1_mean": float(np.mean(cv_res["test_f1"])),
        })

        # Collect feature importances if the classifier exposes them
        # We look at the fitted estimators from each fold
        importances_per_fold = []
        for est in cv_res["estimator"]:
            clf = est.named_steps.get("clf", None)
            if hasattr(clf, "feature_importances_"):
                importances_per_fold.append(clf.feature_importances_)

        if importances_per_fold:
            avg_imp = np.mean(np.vstack(importances_per_fold), axis=0)
            for feat, imp in zip(feature_names, avg_imp):
                feat_imps_rows.append({"Model": name, "Feature": feat, "Importance": float(imp)})

    results_df = pd.DataFrame(results_rows).sort_values(by="CV_F1_mean", ascending=False)
    results_df.to_csv(out_dir / "results.csv", index=False)

    if feat_imps_rows:
        pd.DataFrame(feat_imps_rows).to_csv(out_dir / "feature_importances.csv", index=False)

    # Pick best by mean F1
    best_name = results_df.iloc[0]["Model"]
    best_f1 = results_df.iloc[0]["CV_F1_mean"]
    best_pipe = models[best_name]

    # Out-of-fold predictions for confusion matrix
    y_oof = cross_val_predict(best_pipe, X, y, cv=skf)
    cm = confusion_matrix(y, y_oof, labels=[0,1])

    # Plot and save confusion matrix
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix (OOF) - {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["miss (0)","made (1)"]); ax.set_yticklabels(["miss (0)","made (1)"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(plots_dir / "confusion_matrix.png")
    plt.close(fig)

    # Fit best model on ALL data and save bundle
    best_pipe.fit(X, y)
    bundle = {
        "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model_name": best_name,
        "cv_f1_made": float(best_f1),
        "pipeline": best_pipe,                 # includes scaler + clf
        "feature_names": feature_names,
        "n_splits": args.n_splits,
        "results_table": results_df.to_dict(orient="list"),
    }
    joblib.dump(bundle, out_dir / "best_phase_model.joblib")

    # Console summary
    print("=== CV Results (mean over folds) ===")
    print(results_df.round(3).to_string(index=False))
    print(f"\n⭐ Best by CV F1(made): {best_name} (F1={best_f1:.3f})")
    print(f"💾 Saved bundle to: {out_dir / 'best_phase_model.joblib'}")
    print(f"🖼️ Saved confusion matrix: {plots_dir / 'confusion_matrix.png'}")

if __name__ == "__main__":
    main()
