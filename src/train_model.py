"""
Offline training script for the no-show prediction model.

Run this once (python -m src.train_model) to regenerate the model artifact in
models/. The Streamlit app trains the same pipeline itself on first launch
(cached), so running this script is optional - it exists mainly so the
model + metrics can be inspected/reproduced outside of Streamlit, and so a
pre-trained artifact can be committed for faster app cold-starts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

from src.features import MODEL_FEATURES, clean_and_engineer, load_raw_data, time_based_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "appointments.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def build_candidate_models(pos_weight: float):
    """Three candidate models spanning a simple-to-strong spectrum.

    We deliberately compare a linear baseline against two tree ensembles so
    the final choice is justified by evaluation numbers rather than assumed.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", n_jobs=None
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=8,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42,
            # class imbalance is ~4:1 (No:Yes) -> upweight the minority class
            class_weight="balanced",
        ),
    }


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }


def main():
    print(f"Loading data from {DATA_PATH} ...")
    raw = load_raw_data(str(DATA_PATH))
    df = clean_and_engineer(raw)

    train_df, test_df, cutoff_date = time_based_split(df, test_frac=0.2)
    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows | cutoff: {cutoff_date.date()}")

    X_train, y_train = train_df[MODEL_FEATURES], train_df["no_show_flag"]
    X_test, y_test = test_df[MODEL_FEATURES], test_df["no_show_flag"]

    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    results = {}
    fitted = {}
    for name, model in build_candidate_models(pos_weight).items():
        t0 = time.time()
        if name == "Logistic Regression":
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            model.fit(X_train_s, y_train)
            metrics = evaluate(model, X_test_s, y_test)
            fitted[name] = (model, scaler)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate(model, X_test, y_test)
            fitted[name] = (model, None)
        metrics["train_seconds"] = round(time.time() - t0, 2)
        results[name] = metrics
        print(f"  {name:<24} ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  ({metrics['train_seconds']}s)")

    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    best_model, best_scaler = fitted[best_name]
    print(f"\nBest model by ROC-AUC: {best_name}")

    # Feature importance for the champion model
    if best_scaler is not None:
        importance = permutation_importance(
            best_model, best_scaler.transform(X_test), y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
    else:
        importance = permutation_importance(
            best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
        )
    importance_df = pd.DataFrame(
        {"feature": MODEL_FEATURES, "importance": importance.importances_mean}
    ).sort_values("importance", ascending=False)

    joblib.dump(
        {"model": best_model, "scaler": best_scaler, "model_name": best_name, "features": MODEL_FEATURES},
        MODEL_DIR / "model.joblib",
    )
    importance_df.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    summary = {
        "best_model": best_name,
        "cutoff_date": str(cutoff_date.date()),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "all_results": {
            name: {"roc_auc": r["roc_auc"], "pr_auc": r["pr_auc"], "train_seconds": r["train_seconds"]}
            for name, r in results.items()
        },
        "best_model_full_metrics": results[best_name],
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved model -> {MODEL_DIR/'model.joblib'}")
    print(f"Saved metrics -> {MODEL_DIR/'metrics.json'}")


if __name__ == "__main__":
    main()
