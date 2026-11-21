"""Evaluation metrics. Every value is computed with scikit-learn from held-out predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import metrics


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, classes: list[str]
) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            metrics.precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            metrics.recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": None,
        "log_loss": None,
    }
    if proba is not None and len(set(y_true)) > 1:
        if len(classes) == 2:
            out["roc_auc"] = float(metrics.roc_auc_score(y_true == classes[1], proba[:, 1]))
        elif len(set(y_true)) == len(classes):
            out["roc_auc"] = float(
                metrics.roc_auc_score(
                    y_true, proba, multi_class="ovr", average="macro", labels=classes
                )
            )
        out["log_loss"] = float(metrics.log_loss(y_true, proba, labels=classes))
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict[str, Any]:
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=classes)
    return {"labels": classes, "matrix": matrix.tolist()}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {
        "mae": float(metrics.mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(y_true, y_pred))),
        "r2": float(metrics.r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
        "median_ae": float(metrics.median_absolute_error(y_true, y_pred)),
        "mape": None,
    }
    if np.all(y_true != 0):  # MAPE is undefined when any true value is zero
        out["mape"] = float(metrics.mean_absolute_percentage_error(y_true, y_pred))
    return out


# scikit-learn scorer names used for cross-validation and permutation importance
CV_SCORERS = {
    "classification": {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
    },
    "regression": {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    },
}
PRIMARY_METRIC = {"classification": "balanced_accuracy", "regression": "r2"}
