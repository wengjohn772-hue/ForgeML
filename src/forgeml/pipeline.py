"""The scikit-learn pipeline: preprocessing learned on training data only, then the model."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forgeml.data import FeatureSet
from forgeml.models import build_estimator
from forgeml.spec import RunSpec


def build_preprocessor(features: FeatureSet, spec: RunSpec) -> ColumnTransformer:
    p = spec.preprocessing
    transformers: list[tuple[str, Any, list[str]]] = []
    if features.numeric:
        steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy=p.numeric_impute))]
        if p.scale_numeric:
            steps.append(("scale", StandardScaler()))
        transformers.append(("numeric", Pipeline(steps), features.numeric))
    if features.categorical:
        impute = (
            SimpleImputer(strategy="most_frequent")
            if p.categorical_impute == "most_frequent"
            else SimpleImputer(strategy="constant", fill_value="__missing__")
        )
        encode = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            max_categories=p.max_categories,
            sparse_output=False,  # every supported model accepts dense input
        )
        transformers.append(
            (
                "categorical",
                Pipeline([("impute", impute), ("encode", encode)]),
                features.categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def build_pipeline(features: FeatureSet, spec: RunSpec) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", build_preprocessor(features, spec)),
            ("model", build_estimator(spec.model.name, spec.model.params, spec.seed)),
        ]
    )
