"""Training/evaluation behaviour, determinism, safety and the command line."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from forgeml.errors import ForgeMLError
from forgeml.persistence import load_model, save_model
from forgeml.runner import run
from forgeml.spec import RunSpec
from tests.conftest import churn_spec, price_spec, write_spec


def run_spec(directory: Path, spec: dict[str, Any], out: str = "run") -> dict[str, Any]:
    return run(RunSpec.model_validate(spec), base_dir=directory, out_dir=directory / out)


# ---------- spec validation ----------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": {"name": "xgboost_magic"}}, "unknown model"),
        ({"model": {"name": "ridge"}}, "is for regression"),
        (
            {"model": {"name": "random_forest_classifier", "params": {"n_estimators": 5}}},
            "greater than or equal to 10",
        ),
        ({"model": {"name": "logistic_regression", "params": {"penalty": "l1"}}}, "Extra inputs"),
        ({"exclude": ["churned"]}, "cannot also be a feature"),
        ({"surprise": True}, "Extra inputs"),
        ({"split": {"test_size": 0.9}}, "less than or equal to 0.5"),
    ],
)
def test_invalid_specs_are_rejected_before_training(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        RunSpec.model_validate(churn_spec(**change))


def test_defaults_are_filled_in() -> None:
    spec = RunSpec.model_validate(churn_spec(model={"name": "random_forest_classifier"}))
    assert spec.model.params == {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_leaf": 1,
        "class_weight": None,
    }
    assert spec.split.test_size == 0.2 and spec.seed == 42


# ---------- classification ----------


def test_classification_learns_the_planted_signal(churn_dir: Path) -> None:
    result = run_spec(churn_dir, churn_spec())
    test, baseline = result["metrics"]["test"], result["metrics"]["baseline"]
    assert test["balanced_accuracy"] > baseline["balanced_accuracy"] + 0.15
    assert test["roc_auc"] > 0.8 and baseline["roc_auc"] == 0.5
    assert result["features"]["numeric"] == ["tenure_months", "monthly_spend"]
    assert result["features"]["categorical"] == ["plan"]
    reasons = {e["name"]: e["reason"] for e in result["features"]["excluded"]}
    assert reasons == {
        "customer_id": "identifier-like (every value unique)",
        "signup": "date/time columns are not used",
    }
    assert result["feature_importance"]["features"][0]["feature"] == "tenure_months"
    matrix = result["confusion_matrix"]["matrix"]
    assert result["confusion_matrix"]["labels"] == ["no", "yes"]
    assert sum(map(sum, matrix)) == result["data"]["test_rows"] == 120
    assert set(result["metrics"]["cv"]["metrics"]) == {"accuracy", "f1_macro", "balanced_accuracy"}
    assert len(result["metrics"]["cv"]["metrics"]["accuracy"]["folds"]) == 3
    predictions = pl.read_csv(churn_dir / "run" / "predictions.csv")
    assert predictions.columns == ["row", "actual", "predicted", "proba_no", "proba_yes"]
    assert predictions.height == 120


def test_runs_are_deterministic(churn_dir: Path) -> None:
    volatile = {"created_at", "duration_seconds"}
    first = run_spec(
        churn_dir,
        churn_spec(model={"name": "random_forest_classifier", "params": {"n_estimators": 50}}),
        "a",
    )
    second = run_spec(
        churn_dir,
        churn_spec(model={"name": "random_forest_classifier", "params": {"n_estimators": 50}}),
        "b",
    )
    assert {k: v for k, v in first.items() if k not in volatile} == {
        k: v for k, v in second.items() if k not in volatile
    }
    assert (churn_dir / "a" / "predictions.csv").read_text() == (
        churn_dir / "b" / "predictions.csv"
    ).read_text()


def test_missing_targets_are_dropped_and_counted(tmp_path: Path) -> None:
    from tests.conftest import churn_frame

    df = churn_frame(200)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) % 10 == 0)
        .then(None)
        .otherwise(pl.col("churned"))
        .alias("churned")
    )
    df.write_csv(tmp_path / "customers.csv")
    result = run_spec(tmp_path, churn_spec(model={"name": "logistic_regression"}))
    assert result["data"]["rows"] == 200
    assert result["data"]["rows_dropped_missing_target"] == 20
    assert result["data"]["train_rows"] + result["data"]["test_rows"] == 180


def test_multiclass(tmp_path: Path) -> None:
    from tests.conftest import churn_frame

    churn_frame(300).with_columns(
        pl.when(pl.col("tenure_months") < 20)
        .then(pl.lit("new"))
        .when(pl.col("tenure_months") < 40)
        .then(pl.lit("settled"))
        .otherwise(pl.lit("loyal"))
        .alias("segment")
    ).drop("churned").write_csv(tmp_path / "customers.csv")
    result = run_spec(tmp_path, churn_spec(target="segment", model={"name": "logistic_regression"}))
    assert result["classes"] == ["loyal", "new", "settled"]
    assert result["metrics"]["test"]["accuracy"] > 0.9  # segment is a function of tenure
    assert result["metrics"]["test"]["roc_auc"] is not None


# ---------- regression ----------


def test_regression_recovers_the_linear_signal(price_dir: Path) -> None:
    result = run_spec(price_dir, price_spec())
    test, baseline = result["metrics"]["test"], result["metrics"]["baseline"]
    assert test["r2"] > 0.9 and baseline["r2"] < 0.05
    assert test["mae"] < baseline["mae"] / 3
    assert result["confusion_matrix"] is None and result["classes"] is None
    assert result["feature_importance"]["features"][0]["feature"] == "size_m2"


def test_regression_needs_a_numeric_target(churn_dir: Path) -> None:
    with pytest.raises(ForgeMLError) as info:
        run_spec(churn_dir, price_spec(target="plan", dataset={"path": "customers.csv"}))
    assert info.value.code == "target_not_numeric"


# ---------- safety ----------


class NotSklearn:
    def __init__(self) -> None:
        self.payload = "anything"


def test_only_sklearn_and_numpy_types_are_trusted(tmp_path: Path) -> None:
    path = tmp_path / "evil.skops"
    save_model({"model": NotSklearn()}, path)
    with pytest.raises(ForgeMLError) as info:
        load_model(path)
    assert info.value.code == "untrusted_model"


def test_saved_model_loads_and_predicts_identically(churn_dir: Path) -> None:
    run_spec(churn_dir, churn_spec())
    out = churn_dir / "scored.csv"
    from forgeml.runner import predict

    predict(churn_dir / "run", churn_dir / "customers.csv", out)
    scored = pl.read_csv(out)
    held_out = pl.read_csv(churn_dir / "run" / "predictions.csv")
    assert scored.height == 600
    rows = held_out.get_column("row").to_list()
    assert (
        scored[rows].get_column("predicted").to_list() == held_out.get_column("predicted").to_list()
    )


# ---------- command line ----------


def test_cli_run_and_errors(churn_dir: Path) -> None:
    spec = write_spec(churn_dir, churn_spec())
    ok = subprocess.run(
        [sys.executable, "-m", "forgeml.cli", "run", str(spec), "--out", str(churn_dir / "cli")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok.returncode == 0, ok.stderr
    assert "test balanced_accuracy" in ok.stdout
    for name in (
        "result.json",
        "model.skops",
        "model.json",
        "predictions.csv",
        "spec.json",
        "reproduce.py",
        "notebook.ipynb",
        "evaluate.R",
    ):
        assert (churn_dir / "cli" / name).exists(), name
    bad = write_spec(churn_dir, churn_spec(target="nope"))
    failed = subprocess.run(
        [sys.executable, "-m", "forgeml.cli", "run", str(bad), "--out", str(churn_dir / "x")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 2 and "target_not_found" in failed.stderr
    assert (
        json.loads((churn_dir / "cli" / "result.json").read_text())["artifacts"]["evaluate_r"]
        == "evaluate.R"
    )
