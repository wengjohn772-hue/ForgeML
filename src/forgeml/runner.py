"""`run(spec)`: load -> split -> fit -> evaluate (vs. baseline) -> explain -> save.

Determinism: the split, every estimator, cross-validation folds and permutation importance use
the spec seed; tree ensembles run single-threaded and BLAS is limited to one thread while
fitting/scoring, so the same spec + data + library versions produce identical results.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from threadpoolctl import threadpool_limits

from forgeml import __version__
from forgeml.data import (
    load_table,
    resolve_format,
    select_features,
    sha256_file,
    target_values,
    to_frame,
)
from forgeml.errors import ForgeMLError
from forgeml.evaluate import (
    CV_SCORERS,
    PRIMARY_METRIC,
    classification_metrics,
    confusion,
    regression_metrics,
)
from forgeml.models import MODELS
from forgeml.persistence import save_model
from forgeml.pipeline import build_pipeline
from forgeml.spec import RunSpec

RESULT_SCHEMA = "forgeml.result/v1"


def environment() -> dict[str, str]:
    packages = ("scikit-learn", "numpy", "pandas", "polars", "skops")
    return {
        "python": platform.python_version(),
        "forgeml": __version__,
        **{name: version(name) for name in packages},
        "machine": platform.machine(),
    }


def _split(y: np.ndarray, spec: RunSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    warnings: list[str] = []
    indices = np.arange(len(y))
    stratify = None
    if spec.task == "classification" and spec.split.stratify:
        _, counts = np.unique(y, return_counts=True)
        test_rows = int(np.ceil(len(y) * spec.split.test_size))
        if counts.min() >= 2 and test_rows >= len(counts):
            stratify = y
        else:
            warnings.append(
                "Stratified split was not possible (a class has fewer than 2 rows); "
                "used a random split."
            )
    train, test = train_test_split(
        indices, test_size=spec.split.test_size, random_state=spec.seed, stratify=stratify
    )
    return np.sort(train), np.sort(test), warnings


def run(spec: RunSpec, *, base_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Execute one run and write result.json, model.skops, predictions.csv and spec.json."""
    started = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = (base_dir / spec.dataset.path).resolve()
    file_format = resolve_format(data_path, spec.dataset.format)
    sha256 = sha256_file(data_path) if data_path.exists() else None
    df = load_table(data_path, file_format)

    if spec.target not in df.columns:
        raise ForgeMLError("target_not_found", f"Target column '{spec.target}' is not in the data.")
    rows_total = df.height
    df = df.with_row_index("__row__").filter(pl.col(spec.target).is_not_null())
    if df.height < 10:
        raise ForgeMLError("too_few_rows", "At least 10 rows with a target value are needed.")
    features = select_features(df.drop("__row__"), spec)
    X, y = to_frame(df, features), target_values(df, spec)
    row_ids = df.get_column("__row__").to_numpy()

    classes: list[str] = []
    if spec.task == "classification":
        classes = sorted(str(c) for c in np.unique(y))
        if len(classes) < 2:
            raise ForgeMLError("single_class", "The target has only one class.")
        if len(classes) > 100:
            raise ForgeMLError(
                "too_many_classes", "More than 100 classes; is this really a category?"
            )

    train, test, warnings = _split(y, spec)
    X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y[train], y[test]
    pipeline = build_pipeline(features, spec)
    is_clf = spec.task == "classification"

    with threadpool_limits(limits=1):
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        proba = (
            pipeline.predict_proba(X_test)
            if is_clf and hasattr(pipeline, "predict_proba")
            else None
        )

        baseline = DummyClassifier(strategy="prior") if is_clf else DummyRegressor(strategy="mean")
        baseline.fit(X_train, y_train)
        b_pred = baseline.predict(X_test)
        b_proba = baseline.predict_proba(X_test) if is_clf else None

        if is_clf:
            test_metrics = classification_metrics(y_test, y_pred, proba, classes)
            baseline_metrics = classification_metrics(y_test, b_pred, b_proba, classes)
        else:
            test_metrics = regression_metrics(y_test, y_pred)
            baseline_metrics = regression_metrics(y_test, b_pred)

        cv: dict[str, Any] | None = None
        if spec.cv_folds:
            splitter = (
                StratifiedKFold(spec.cv_folds, shuffle=True, random_state=spec.seed)
                if is_clf and "Stratified" not in " ".join(warnings)
                else KFold(spec.cv_folds, shuffle=True, random_state=spec.seed)
            )
            scorers = CV_SCORERS[spec.task]
            scores = cross_validate(
                clone(pipeline), X_train, y_train, cv=splitter, scoring=scorers, n_jobs=1
            )
            cv = {"folds": spec.cv_folds, "metrics": {}}
            for name, scorer in scorers.items():
                values = scores[f"test_{name}"] * (-1 if scorer.startswith("neg_") else 1)
                cv["metrics"][name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "folds": [float(v) for v in values],
                }

        primary = PRIMARY_METRIC[spec.task]
        importance = permutation_importance(
            pipeline,
            X_test,
            y_test,
            scoring=CV_SCORERS[spec.task][primary],
            n_repeats=spec.permutation_repeats,
            random_state=spec.seed,
            n_jobs=1,
        )
    triples = zip(
        features.all, importance.importances_mean, importance.importances_std, strict=True
    )
    ordered = sorted(
        ((name, float(m), float(s)) for name, m, s in triples), key=lambda x: (-x[1], x[0])
    )
    ranked = [{"feature": name, "mean": m, "std": s} for name, m, s in ordered]

    # predictions on the held-out test set
    predictions: dict[str, Any] = {
        "row": row_ids[test],
        "actual": y_test if is_clf else y_test.astype(float),
        "predicted": y_pred if is_clf else y_pred.astype(float),
    }
    if proba is not None:
        for i, label in enumerate(classes):
            predictions[f"proba_{label}"] = proba[:, i]
    pl.DataFrame(predictions).write_csv(out_dir / "predictions.csv")
    save_model(pipeline, out_dir / "model.skops")
    (out_dir / "spec.json").write_text(json.dumps(spec.to_json_dict(), indent=2))
    model_card = {
        "task": spec.task,
        "target": spec.target,
        "features": {"numeric": features.numeric, "categorical": features.categorical},
        "classes": classes or None,
        "forgeml": __version__,
    }
    (out_dir / "model.json").write_text(json.dumps(model_card, indent=2))

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment(),
        "spec": spec.to_json_dict(),
        "model": {
            "name": spec.model.name,
            "label": MODELS[spec.model.name].label,
            "params": spec.model.params,
        },
        "data": {
            "sha256": sha256,
            "format": file_format,
            "rows": rows_total,
            "rows_used": df.height,
            "rows_dropped_missing_target": rows_total - df.height,
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "features": {
            "numeric": features.numeric,
            "categorical": features.categorical,
            "excluded": features.excluded,
        },
        "classes": classes or None,
        "metrics": {
            "primary": primary,
            "test": test_metrics,
            "baseline": baseline_metrics,
            "cv": cv,
        },
        "confusion_matrix": confusion(y_test, y_pred, classes) if is_clf else None,
        "feature_importance": {
            "method": "permutation",
            "scoring": CV_SCORERS[spec.task][primary],
            "features": ranked,
        },
        "artifacts": {
            "model": "model.skops",
            "model_card": "model.json",
            "predictions": "predictions.csv",
            "spec": "spec.json",
        },
        "warnings": warnings,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    return result


def predict(model_dir: Path, data_path: Path, out_path: Path) -> int:
    """Score new rows with a saved run. Returns the number of rows written."""
    from forgeml.data import FeatureSet
    from forgeml.persistence import load_model

    card = json.loads((model_dir / "model.json").read_text())
    pipeline = load_model(model_dir / "model.skops")
    df = load_table(data_path, resolve_format(data_path, None))
    features = FeatureSet(
        numeric=card["features"]["numeric"], categorical=card["features"]["categorical"]
    )
    missing = [c for c in features.all if c not in df.columns]
    if missing:
        raise ForgeMLError(
            "column_not_found", f"Columns needed by the model are missing: {missing}"
        )
    X = to_frame(df, features)
    with threadpool_limits(limits=1):
        out: dict[str, Any] = {"predicted": pipeline.predict(X)}
        if card["task"] == "classification" and hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X)
            for i, label in enumerate(card["classes"]):
                out[f"proba_{label}"] = proba[:, i]
    pl.DataFrame(out).write_csv(out_path)
    return df.height
