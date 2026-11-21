"""The exported reproduce.py, notebook and evaluate.R must confirm the original results."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from tests.conftest import churn_spec, price_spec, write_spec


def cli_run(directory: Path, spec: dict, out: str = "run") -> Path:  # type: ignore[type-arg]
    spec_path = write_spec(directory, spec)
    subprocess.run(
        [sys.executable, "-m", "forgeml.cli", "run", str(spec_path), "--out", str(directory / out)],
        check=True,
        capture_output=True,
    )
    run_dir = directory / out
    # the exports expect the dataset next to them, like a downloaded run folder
    shutil.copy(directory / spec["dataset"]["path"], run_dir / spec["dataset"]["path"])
    return run_dir


def test_reproduce_script_confirms_every_number(churn_dir: Path) -> None:
    run_dir = cli_run(churn_dir, churn_spec())
    result = subprocess.run(
        [sys.executable, str(run_dir / "reproduce.py")],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Reproduced exactly." in result.stdout
    assert "DIFF" not in result.stdout


def test_reproduce_script_detects_changed_data(churn_dir: Path) -> None:
    run_dir = cli_run(churn_dir, churn_spec())
    data = run_dir / "customers.csv"
    data.write_text(data.read_text().replace("basic", "pro", 30))
    result = subprocess.run(
        [sys.executable, str(run_dir / "reproduce.py")],
        cwd=run_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "DIFF   data sha256" in result.stdout


def test_notebook_executes_and_matches(price_dir: Path) -> None:
    run_dir = cli_run(price_dir, price_spec())
    notebook = nbformat.read(run_dir / "notebook.ipynb", as_version=4)
    NotebookClient(
        notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(run_dir)}}
    ).execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    assert "Reproduced exactly." in "".join(o.get("text", "") for o in outputs)
    assert sum("image/png" in o.get("data", {}) for o in outputs) == 2


r_available = (
    shutil.which("Rscript") is not None
    and subprocess.run(
        ["Rscript", "-e", "quit(status = !requireNamespace('jsonlite', quietly = TRUE))"],
        capture_output=True,
        check=False,
    ).returncode
    == 0
    if shutil.which("Rscript")
    else False
)


@pytest.mark.skipif(not r_available, reason="R with jsonlite is not installed")
@pytest.mark.parametrize("kind", ["binary", "multiclass", "regression"])
def test_r_recomputes_the_same_metrics(tmp_path: Path, kind: str) -> None:
    import polars as pl

    from tests.conftest import churn_frame, price_frame

    if kind == "regression":
        price_frame().write_csv(tmp_path / "houses.csv")
        spec = price_spec(model={"name": "random_forest_regressor", "params": {"n_estimators": 30}})
    else:
        frame = churn_frame()
        if kind == "multiclass":
            frame = frame.with_columns(
                pl.when(pl.col("tenure_months") < 20)
                .then(pl.lit("new"))
                .when(pl.col("tenure_months") < 40)
                .then(pl.lit("settled"))
                .otherwise(pl.lit("loyal"))
                .alias("churned")
            )
        frame.write_csv(tmp_path / "customers.csv")
        spec = churn_spec(model={"name": "logistic_regression"})
    run_dir = cli_run(tmp_path, spec)
    result = subprocess.run(
        ["Rscript", "evaluate.R"], cwd=run_dir, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All metrics match." in result.stdout
