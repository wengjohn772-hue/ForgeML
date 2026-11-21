from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest


def churn_frame(n: int = 600, seed: int = 0) -> pl.DataFrame:
    """Churn depends on tenure (strongly) and plan (moderately); the rest is noise."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 60, n)
    plan = rng.choice(["basic", "pro", "team"], n)
    spend = np.round(rng.normal(80, 25, n), 2)
    logit = 2.0 - 0.09 * tenure + np.where(plan == "basic", 0.8, -0.4) + rng.normal(0, 0.6, n)
    churned = np.where(1 / (1 + np.exp(-logit)) > 0.5, "yes", "no")
    spend_with_gaps = [None if i % 23 == 0 else float(v) for i, v in enumerate(spend)]
    return pl.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],  # identifier: auto-excluded
            "tenure_months": tenure,
            "plan": plan,
            "monthly_spend": spend_with_gaps,  # has missing values
            "signup": [
                f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)
            ],  # date: excluded
            "churned": churned,
        }
    )


def price_frame(n: int = 400, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    size = rng.uniform(30, 200, n)
    rooms = rng.integers(1, 6, n)
    area = rng.choice(["north", "south", "east"], n)
    price = (
        1500 * size + 8000 * rooms + np.where(area == "north", 40000, 0) + rng.normal(0, 15000, n)
    )
    return pl.DataFrame(
        {"size_m2": np.round(size, 1), "rooms": rooms, "area": area, "price": np.round(price, 2)}
    )


@pytest.fixture
def churn_dir(tmp_path: Path) -> Path:
    churn_frame().write_csv(tmp_path / "customers.csv")
    return tmp_path


@pytest.fixture
def price_dir(tmp_path: Path) -> Path:
    price_frame().write_csv(tmp_path / "houses.csv")
    return tmp_path


def churn_spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "schema": "forgeml.spec/v1",
        "task": "classification",
        "target": "churned",
        "dataset": {"path": "customers.csv"},
        "model": {"name": "gradient_boosting_classifier", "params": {"max_iter": 60}},
        "cv_folds": 3,
        "permutation_repeats": 3,
    }
    spec.update(overrides)
    return spec


def price_spec(**overrides: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "task": "regression",
        "target": "price",
        "dataset": {"path": "houses.csv"},
        "model": {"name": "ridge"},
        "cv_folds": 3,
    }
    spec.update(overrides)
    return spec


def write_spec(directory: Path, spec: dict[str, Any]) -> Path:
    path = directory / "spec.json"
    path.write_text(json.dumps(spec))
    return path
