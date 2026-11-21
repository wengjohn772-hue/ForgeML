"""The run specification (`forgeml.spec/v1`): everything that determines a run's result.

A spec plus the same data file and the same library versions reproduces a run exactly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forgeml.models import MODELS, validate_params

SPEC_SCHEMA = "forgeml.spec/v1"
Task = Literal["classification", "regression"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetSpec(_Strict):
    path: str = Field(min_length=1, description="CSV, TSV or Parquet file, relative to the spec")
    format: Literal["csv", "tsv", "parquet"] | None = Field(
        None, description="inferred from the file extension when omitted"
    )


class PreprocessingSpec(_Strict):
    numeric_impute: Literal["mean", "median"] = "median"
    scale_numeric: bool = True
    categorical_impute: Literal["most_frequent", "constant"] = "most_frequent"
    max_categories: int = Field(50, ge=2, le=500, description="rarer categories are grouped")


class SplitSpec(_Strict):
    test_size: float = Field(0.2, gt=0.0, le=0.5)
    stratify: bool = Field(True, description="classification only; keeps class proportions")


class ModelSpec(_Strict):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class RunSpec(_Strict):
    schema_: Literal["forgeml.spec/v1"] = Field("forgeml.spec/v1", alias="schema")
    task: Task
    target: str = Field(min_length=1)
    dataset: DatasetSpec
    features: list[str] | None = Field(None, description="default: every column except target")
    exclude: list[str] = Field(default_factory=list)
    preprocessing: PreprocessingSpec = Field(default_factory=PreprocessingSpec)
    split: SplitSpec = Field(default_factory=SplitSpec)
    model: ModelSpec
    cv_folds: int | None = Field(5, ge=2, le=10, description="cross-validation on the training set")
    permutation_repeats: int = Field(5, ge=1, le=50)
    seed: int = Field(42, ge=0, le=2**32 - 1)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _check(self) -> RunSpec:
        entry = MODELS.get(self.model.name)
        if entry is None:
            raise ValueError(f"unknown model '{self.model.name}'; available: {sorted(MODELS)}")
        if entry.task != self.task:
            raise ValueError(f"model '{self.model.name}' is for {entry.task}, not {self.task}")
        self.model.params = validate_params(self.model.name, self.model.params)
        if self.target in self.exclude or (self.features and self.target in self.features):
            raise ValueError("the target cannot also be a feature or excluded")
        return self

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True)
