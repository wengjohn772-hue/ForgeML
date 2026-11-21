# ForgeML

**Reproducible tabular machine learning from a single JSON spec.**

ForgeML trains and evaluates scikit-learn models on CSV, TSV or Parquet data and writes
everything needed to trust, explain and reproduce the result:

- a held-out evaluation **compared against a naive baseline**, so a weak model cannot look good;
- cross-validation on the training set;
- permutation feature importance;
- the fitted model in the [skops](https://skops.readthedocs.io) format, which, unlike pickle, cannot run code when loaded;
- a `reproduce.py` script and a notebook that re-run the spec and check every number, plus an
  `evaluate.R` script that recomputes the metrics independently in R.

ForgeML is a standalone tool. It also powers the experiments feature of the XModula platform,
which uses only its public spec/result contract.

## Quick start

```bash
pip install "forgeml @ git+https://github.com/wengjohn772-hue/ForgeML@v0.1.1"
forgeml run examples/churn.json --out run/
```

```text
Gradient boosting (histogram) · classification · target 'churned'
test balanced_accuracy: 0.8995   (baseline 0.5000)
results written to run/
```

(`examples/customers.csv` is synthetic: churn depends on tenure and plan, plus noise.)

## The spec (`forgeml.spec/v1`)

```json
{
  "schema": "forgeml.spec/v1",
  "task": "classification",
  "target": "churned",
  "dataset": { "path": "customers.csv" },
  "exclude": ["customer_id"],
  "preprocessing": { "numeric_impute": "median", "scale_numeric": true, "max_categories": 50 },
  "split": { "test_size": 0.2, "stratify": true },
  "model": { "name": "gradient_boosting_classifier", "params": { "learning_rate": 0.1 } },
  "cv_folds": 5,
  "seed": 42
}
```

Only `task`, `target`, `dataset` and `model` are required. Every field is validated before any
training starts (`forgeml models` lists the models and their parameters).

| Task | Models |
|---|---|
| classification | `logistic_regression`, `random_forest_classifier`, `gradient_boosting_classifier` |
| regression | `linear_regression`, `ridge`, `random_forest_regressor`, `gradient_boosting_regressor` |

**Feature handling.** All columns except the target are used unless you pass `features` or
`exclude`. Numeric columns are imputed and (optionally) scaled. Text and boolean columns are
imputed and one-hot encoded; rare categories are grouped. Preprocessing is learned on the
training split only. Some columns are excluded automatically, and each exclusion is listed with
its reason in the result:

- constant columns;
- date/time columns;
- identifier-like columns, where every value is unique.

Rows with a missing target are dropped and counted.

## Outputs

| File | Contents |
|---|---|
| `result.json` | `forgeml.result/v1`: environment, spec, data SHA-256, features, test/baseline/CV metrics, confusion matrix, feature importance, warnings |
| `model.skops`, `model.json` | fitted pipeline (preprocessing + model) and its feature list |
| `predictions.csv` | test-set rows: original row number, actual, predicted, class probabilities |
| `reproduce.py`, `notebook.ipynb` | re-run the spec and compare every metric (exit code 1 on any difference) |
| `evaluate.R` | independent recomputation of the metrics in base R from `predictions.csv` |

Score new data with a finished run:

```bash
forgeml predict run/ new_customers.csv --out scored.csv
```

## Metrics

- **Classification:** accuracy, balanced accuracy (primary), macro precision/recall/F1, weighted
  F1, ROC-AUC (binary, or one-vs-rest macro) and log loss. The baseline predicts class priors.
- **Regression:** MAE, RMSE, R² (primary), median absolute error and MAPE. MAPE is reported only
  when no true value is zero. The baseline predicts the training mean.

## Reproducibility

The same spec, data file and library versions produce **identical** results. ForgeML makes this
hold by:

- using the spec seed for the split, every estimator, the CV folds and permutation importance;
- training tree ensembles single-threaded, and limiting BLAS to one thread while fitting and
  scoring;
- recording the environment and the data SHA-256 in `result.json`.

Results across different CPU architectures can differ in the last floating-point digits.
`reproduce.py` reports exactly which parts match.

## Security

- Model files use skops. When loading, ForgeML trusts only scikit-learn and NumPy types and
  refuses anything else.
- The spec cannot name arbitrary Python objects: models and parameters come from a fixed,
  validated registry.

## Development

```bash
uv sync
uv run pytest          # includes notebook execution and, when R is installed, the R checks
uv run ruff check . && uv run mypy
```

## License

MIT
