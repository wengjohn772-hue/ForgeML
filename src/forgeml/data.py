"""Loading data and choosing features.

Columns are typed by Polars with full-file inference (the same rules as XModula's profiler);
features are then handed to scikit-learn as a pandas frame built column by column.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from forgeml.errors import ForgeMLError
from forgeml.spec import RunSpec

NULL_TOKENS = ["", "NA", "N/A", "n/a", "NaN", "nan", "null", "NULL", "None", "none", "-"]
IDENTIFIER_MIN_ROWS = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_format(path: Path, declared: str | None) -> str:
    if declared:
        return declared
    suffix = path.suffix.lower()
    formats = {".csv": "csv", ".tsv": "tsv", ".tab": "tsv", ".parquet": "parquet"}
    if suffix not in formats:
        raise ForgeMLError("unsupported_format", f"Cannot infer the format of '{path.name}'.")
    return formats[suffix]


def load_table(path: Path, file_format: str) -> pl.DataFrame:
    if not path.exists():
        raise ForgeMLError("file_not_found", f"Dataset file not found: {path}")
    try:
        if file_format == "parquet":
            return pl.read_parquet(path)
        df = pl.read_csv(
            path,
            separator="\t" if file_format == "tsv" else ",",
            infer_schema_length=None,
            null_values=NULL_TOKENS,
        )
    except (pl.exceptions.PolarsError, OSError, ValueError) as exc:
        raise ForgeMLError("parse_error", f"Could not read the dataset as {file_format}.") from exc
    return _parse_temporal(df)


def _to_date(s: pl.Series) -> pl.Series:
    return s.str.to_date(strict=False)


def _to_datetime(s: pl.Series) -> pl.Series:
    return s.str.to_datetime(strict=False, time_unit="us")


def _parse_temporal(df: pl.DataFrame) -> pl.DataFrame:
    """Text columns become Date/Datetime only when EVERY non-empty value parses (the same rule
    as XModula's profiler), so no value is ever silently turned into a missing one."""
    converted = []
    for name, dtype in df.schema.items():
        if dtype != pl.String:
            continue
        s = df.get_column(name)
        sample = s.drop_nulls().head(200)
        if sample.len() == 0:
            continue
        for parse in (_to_date, _to_datetime):
            try:
                if parse(sample).null_count() != 0:
                    continue
                parsed = parse(s)
            except pl.exceptions.PolarsError:
                continue
            if parsed.null_count() == s.null_count():
                converted.append(parsed.alias(name))
                break
    return df.with_columns(converted) if converted else df


@dataclass
class FeatureSet:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    excluded: list[dict[str, str]] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return self.numeric + self.categorical


def select_features(df: pl.DataFrame, spec: RunSpec) -> FeatureSet:
    if spec.target not in df.columns:
        raise ForgeMLError("target_not_found", f"Target column '{spec.target}' is not in the data.")
    candidates = spec.features or [c for c in df.columns if c != spec.target]
    missing = [c for c in candidates + spec.exclude if c not in df.columns]
    if missing:
        raise ForgeMLError("column_not_found", f"Columns not in the data: {missing}")

    features = FeatureSet()
    for name in candidates:
        s = df.get_column(name)
        non_null = s.len() - s.null_count()
        if name in spec.exclude:
            features.excluded.append({"name": name, "reason": "excluded by the spec"})
        elif non_null == 0:
            features.excluded.append({"name": name, "reason": "no values"})
        elif s.dtype.is_temporal():
            features.excluded.append({"name": name, "reason": "date/time columns are not used"})
        elif s.drop_nulls().n_unique() <= 1:
            features.excluded.append({"name": name, "reason": "constant"})
        elif s.dtype.is_numeric():
            features.numeric.append(name)
        elif (
            s.dtype == pl.Boolean
            or s.dtype == pl.String
            or isinstance(s.dtype, (pl.Categorical, pl.Enum))
        ):
            if (
                s.dtype != pl.Boolean
                and non_null >= IDENTIFIER_MIN_ROWS
                and s.drop_nulls().n_unique() == non_null
            ):
                features.excluded.append(
                    {"name": name, "reason": "identifier-like (every value unique)"}
                )
            else:
                features.categorical.append(name)
        else:
            features.excluded.append({"name": name, "reason": f"unsupported type {s.dtype}"})
    if not features.all:
        raise ForgeMLError("no_features", "No usable feature columns remain.")
    return features


def to_frame(df: pl.DataFrame, features: FeatureSet) -> pd.DataFrame:
    """pandas frame for scikit-learn: numeric as float64 (NaN for missing), categorical as
    strings (None for missing). Built column by column, without needing pyarrow."""
    columns: dict[str, np.ndarray] = {}
    for name in features.numeric:
        columns[name] = df.get_column(name).cast(pl.Float64).to_numpy()
    for name in features.categorical:
        values = df.get_column(name).cast(pl.String).to_numpy()
        columns[name] = values.astype(object)
    return pd.DataFrame(columns, columns=features.all)


def target_values(df: pl.DataFrame, spec: RunSpec) -> np.ndarray:
    s = df.get_column(spec.target)
    if spec.task == "regression":
        if not s.dtype.is_numeric():
            raise ForgeMLError(
                "target_not_numeric",
                f"Regression needs a numeric target; '{spec.target}' is {s.dtype}.",
            )
        return s.cast(pl.Float64).to_numpy()
    return s.cast(pl.String).to_numpy().astype(object)
