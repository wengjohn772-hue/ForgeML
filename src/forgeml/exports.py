"""Reproducibility exports written next to every run.

- reproduce.py   re-runs the spec with ForgeML and compares every metric and importance with
                 the original result.json (exit code 1 on any difference).
- notebook.ipynb same, with charts (confusion matrix / predicted-vs-actual, importance).
- evaluate.R     independent base-R recomputation of the evaluation metrics from
                 predictions.csv, compared with result.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INSTALL = 'pip install "forgeml @ git+https://github.com/wengjohn772-hue/ForgeML@v{version}"'

COMPARE = """import json
import sys
from pathlib import Path

from forgeml.runner import run
from forgeml.spec import RunSpec

HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


def reproduce(data_file=None):
    original = json.loads((HERE / "result.json").read_text())
    spec = RunSpec.model_validate(json.loads((HERE / "spec.json").read_text()))
    if data_file:
        spec.dataset.path = str(Path(data_file).resolve())
    rerun = run(spec, base_dir=HERE, out_dir=HERE / "reproduced")
    for name, used in original["environment"].items():
        if name not in ("machine",) and rerun["environment"].get(name) != used:
            print(f"note: {name} {rerun['environment'].get(name)} here, {used} in the original run")
    checks = {
        "data sha256": (original["data"]["sha256"], rerun["data"]["sha256"]),
        "test metrics": (original["metrics"]["test"], rerun["metrics"]["test"]),
        "baseline metrics": (original["metrics"]["baseline"], rerun["metrics"]["baseline"]),
        "cross-validation": (original["metrics"]["cv"], rerun["metrics"]["cv"]),
        "confusion matrix": (original["confusion_matrix"], rerun["confusion_matrix"]),
        "feature importance": (original["feature_importance"], rerun["feature_importance"]),
    }
    ok = True
    for label, (a, b) in checks.items():
        same = a == b
        ok &= same
        print(f"{'MATCH' if same else 'DIFF '}  {label}")
    print("Reproduced exactly." if ok else "Results differ — see DIFF lines above.")
    return ok, rerun
"""

# Only the header is formatted; COMPARE contains its own {placeholders} (f-strings).
SCRIPT_HEADER = '''#!/usr/bin/env python3
"""Reproduce ForgeML run: {label} predicting '{target}' ({task}).

Requirements: {install}
              (the original run used: {versions})
Usage:        python reproduce.py [dataset file]
              Keep spec.json and result.json next to this script; the dataset defaults to
              the path in spec.json.
"""
'''

SCRIPT_FOOTER = """

if __name__ == "__main__":
    matched, _ = reproduce(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.exit(0 if matched else 1)
"""

CHARTS = """%matplotlib inline
import matplotlib.pyplot as plt
import polars as pl

predictions = pl.read_csv(HERE / "reproduced" / "predictions.csv")
if rerun["confusion_matrix"]:
    cm = rerun["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(1.5 + 0.6 * len(cm["labels"]), 1.2 + 0.6 * len(cm["labels"])))
    ax.imshow(cm["matrix"], cmap="Blues")
    ax.set_xticks(range(len(cm["labels"])), cm["labels"], rotation=45, ha="right")
    ax.set_yticks(range(len(cm["labels"])), cm["labels"])
    for i, row in enumerate(cm["matrix"]):
        for j, value in enumerate(row):
            ax.text(j, i, value, ha="center", va="center")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title("Confusion matrix (test set)")
    plt.tight_layout(); plt.show()
else:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(predictions["actual"], predictions["predicted"], s=8, alpha=0.6)
    lo = min(predictions["actual"].min(), predictions["predicted"].min())
    hi = max(predictions["actual"].max(), predictions["predicted"].max())
    ax.plot([lo, hi], [lo, hi], color="grey", linestyle="--")
    ax.set_xlabel("actual"); ax.set_ylabel("predicted"); ax.set_title("Predicted vs actual (test set)")
    plt.tight_layout(); plt.show()

top = rerun["feature_importance"]["features"][:15][::-1]
fig, ax = plt.subplots(figsize=(6, 0.35 * len(top) + 1))
ax.barh([f["feature"] for f in top], [f["mean"] for f in top], xerr=[f["std"] for f in top], color="#6366f1")
ax.set_title("Permutation importance (test set)"); ax.set_xlabel(rerun["feature_importance"]["scoring"])
plt.tight_layout(); plt.show()"""

R_SCRIPT = r"""#!/usr/bin/env Rscript
# Independent check of ForgeML run: __LABEL__ predicting '__TARGET__' (__TASK__).
# Recomputes the evaluation metrics in base R from predictions.csv and compares them with
# result.json. This verifies the *evaluation*; training itself happens in Python.
# Requirements: R >= 4.1 and jsonlite (install.packages("jsonlite"))
# Usage: Rscript evaluate.R      (run in the folder with result.json and predictions.csv)

suppressWarnings(suppressMessages(library(jsonlite)))
result <- fromJSON("result.json", simplifyVector = FALSE)
pred <- read.csv("predictions.csv", check.names = FALSE, stringsAsFactors = FALSE,
                 colClasses = c(actual = "character", predicted = "character"))
task <- result$spec$task
tol <- 1e-9
r <- list()

if (task == "classification") {
  y <- pred$actual; p <- pred$predicted
  labels <- sort(unique(c(y, p)), method = "radix")
  per <- lapply(labels, function(l) {
    tp <- sum(y == l & p == l); fp <- sum(y != l & p == l); fn <- sum(y == l & p != l)
    prec <- if (tp + fp == 0) 0 else tp / (tp + fp)
    rec <- if (tp + fn == 0) 0 else tp / (tp + fn)
    f1 <- if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
    list(prec = prec, rec = rec, f1 = f1, support = sum(y == l))
  })
  r$accuracy <- mean(y == p)
  present <- labels %in% unique(y)
  r$balanced_accuracy <- mean(vapply(per[present], function(x) x$rec, numeric(1)))
  r$precision_macro <- mean(vapply(per, function(x) x$prec, numeric(1)))
  r$recall_macro <- mean(vapply(per, function(x) x$rec, numeric(1)))
  r$f1_macro <- mean(vapply(per, function(x) x$f1, numeric(1)))
  sup <- vapply(per, function(x) x$support, numeric(1))
  r$f1_weighted <- sum(vapply(per, function(x) x$f1, numeric(1)) * sup) / sum(sup)
  classes <- unlist(result$classes)
  proba_cols <- paste0("proba_", classes)
  if (all(proba_cols %in% names(pred))) {
    P <- as.matrix(pred[, proba_cols]); colnames(P) <- classes
    auc <- function(score, positive) {      # Mann-Whitney with average ranks (= ROC AUC)
      rk <- rank(score); n1 <- sum(positive); n0 <- sum(!positive)
      (sum(rk[positive]) - n1 * (n1 + 1) / 2) / (n1 * n0)
    }
    r$roc_auc <- if (length(classes) == 2) auc(P[, classes[2]], y == classes[2]) else
      mean(vapply(classes, function(k) auc(P[, k], y == k), numeric(1)))
    eps <- .Machine$double.eps
    pt <- pmin(pmax(P[cbind(seq_along(y), match(y, classes))], eps), 1 - eps)
    r$log_loss <- -mean(log(pt))
  }
} else {
  y <- as.numeric(pred$actual); p <- as.numeric(pred$predicted); e <- y - p
  r$mae <- mean(abs(e)); r$rmse <- sqrt(mean(e^2))
  r$r2 <- 1 - sum(e^2) / sum((y - mean(y))^2)
  r$median_ae <- stats::median(abs(e))
  if (all(y != 0)) r$mape <- mean(abs(e) / pmax(abs(y), .Machine$double.eps))
}

ok <- TRUE
cat(sprintf("%-20s %18s %18s  %s\n", "metric", "R", "result.json", "status"))
for (name in names(r)) {
  expected <- result$metrics$test[[name]]
  if (is.null(expected)) next
  same <- abs(r[[name]] - expected) <= tol * max(1, abs(expected))
  ok <- ok && same
  cat(sprintf("%-20s %18.12f %18.12f  %s\n", name, r[[name]], expected, if (same) "MATCH" else "DIFF"))
}
cat(if (ok) "All metrics match.\n" else "Some metrics differ.\n")
quit(status = if (ok) 0 else 1)
"""


def write_exports(result: dict[str, Any], out_dir: Path) -> None:
    import nbformat
    from nbformat import v4

    from forgeml import __version__

    env = result["environment"]
    versions = ", ".join(f"{k} {v}" for k, v in env.items() if k not in ("python", "machine"))
    install = INSTALL.format(version=__version__)
    meta = {
        "label": result["model"]["label"],
        "target": result["spec"]["target"],
        "task": result["spec"]["task"],
    }
    header = SCRIPT_HEADER.format(install=install, versions=versions, **meta)
    (out_dir / "reproduce.py").write_text(header + COMPARE + SCRIPT_FOOTER)

    cells = [
        v4.new_markdown_cell(
            f"# Reproduce: {meta['label']} predicting `{meta['target']}` ({meta['task']})\n\n"
            f"Re-runs the exact spec with ForgeML and checks every metric against `result.json`.\n\n"
            f"**Requirements:** `{install}` and `matplotlib`. Original environment: {versions}.\n\n"
            "Keep `spec.json`, `result.json` and the dataset file in this folder."
        ),
        v4.new_code_cell(
            COMPARE + "\nmatched, rerun = reproduce()\nassert matched, 'results differ'"
        ),
        v4.new_markdown_cell("## Charts"),
        v4.new_code_cell(CHARTS),
    ]
    notebook = v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    (out_dir / "notebook.ipynb").write_text(nbformat.writes(notebook))

    r = R_SCRIPT.replace("__LABEL__", meta["label"]).replace(
        "__TARGET__", meta["target"].replace("'", "")
    )
    (out_dir / "evaluate.R").write_text(r.replace("__TASK__", meta["task"]))
    exports = {
        "reproduce": "reproduce.py",
        "notebook": "notebook.ipynb",
        "evaluate_r": "evaluate.R",
    }
    result["artifacts"].update(exports)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
