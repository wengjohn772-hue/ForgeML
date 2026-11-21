"""Supported models and their validated hyper-parameters.

Every estimator is created with a fixed `random_state` (from the spec seed) and `n_jobs=1`
where applicable, so training is deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LogisticParams(_Params):
    C: float = Field(1.0, gt=0, le=1e4)
    max_iter: int = Field(1000, ge=100, le=10000)
    class_weight: Literal["balanced"] | None = None


class ForestParams(_Params):
    n_estimators: int = Field(200, ge=10, le=1000)
    max_depth: int | None = Field(None, ge=1, le=100)
    min_samples_leaf: int = Field(1, ge=1, le=100)


class ForestClassifierParams(ForestParams):
    class_weight: Literal["balanced"] | None = None


class BoostingParams(_Params):
    learning_rate: float = Field(0.1, gt=0, le=1)
    max_iter: int = Field(200, ge=10, le=2000)
    max_depth: int | None = Field(None, ge=1, le=50)
    l2_regularization: float = Field(0.0, ge=0, le=100)


class BoostingClassifierParams(BoostingParams):
    class_weight: Literal["balanced"] | None = None


class LinearParams(_Params):
    fit_intercept: bool = True


class RidgeParams(_Params):
    alpha: float = Field(1.0, ge=0, le=1e4)


@dataclass(frozen=True)
class ModelEntry:
    task: Literal["classification", "regression"]
    label: str
    params: type[_Params]
    build: Callable[[dict[str, Any], int], Any]


MODELS: dict[str, ModelEntry] = {
    "logistic_regression": ModelEntry(
        "classification",
        "Logistic regression",
        LogisticParams,
        lambda p, seed: LogisticRegression(random_state=seed, **p),
    ),
    "random_forest_classifier": ModelEntry(
        "classification",
        "Random forest",
        ForestClassifierParams,
        lambda p, seed: RandomForestClassifier(random_state=seed, n_jobs=1, **p),
    ),
    "gradient_boosting_classifier": ModelEntry(
        "classification",
        "Gradient boosting (histogram)",
        BoostingClassifierParams,
        lambda p, seed: HistGradientBoostingClassifier(random_state=seed, **p),
    ),
    "linear_regression": ModelEntry(
        "regression",
        "Linear regression",
        LinearParams,
        lambda p, seed: LinearRegression(**p),
    ),
    "ridge": ModelEntry(
        "regression",
        "Ridge regression",
        RidgeParams,
        lambda p, seed: Ridge(random_state=seed, **p),
    ),
    "random_forest_regressor": ModelEntry(
        "regression",
        "Random forest",
        ForestParams,
        lambda p, seed: RandomForestRegressor(random_state=seed, n_jobs=1, **p),
    ),
    "gradient_boosting_regressor": ModelEntry(
        "regression",
        "Gradient boosting (histogram)",
        BoostingParams,
        lambda p, seed: HistGradientBoostingRegressor(random_state=seed, **p),
    ),
}


def validate_params(name: str, params: dict[str, Any]) -> dict[str, Any]:
    """Return the full, validated parameter set (defaults filled in)."""
    validated: dict[str, Any] = MODELS[name].params.model_validate(params).model_dump()
    return validated


def build_estimator(name: str, params: dict[str, Any], seed: int) -> Any:
    return MODELS[name].build(params, seed)
