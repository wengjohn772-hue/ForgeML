"""Command line: `forgeml run`, `forgeml predict`, `forgeml models`, `forgeml version`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from forgeml import __version__
from forgeml.errors import ForgeMLError
from forgeml.models import MODELS


def _run(args: argparse.Namespace) -> int:
    from forgeml.exports import write_exports
    from forgeml.runner import run
    from forgeml.spec import RunSpec

    spec_path = Path(args.spec)
    spec = RunSpec.model_validate(json.loads(spec_path.read_text()))
    if args.data:  # override the dataset path (e.g. when the file lives elsewhere)
        spec.dataset.path = str(Path(args.data).resolve())
    out = Path(args.out)
    result = run(spec, base_dir=spec_path.parent, out_dir=out)
    if not args.no_exports:
        write_exports(result, out)
    m = result["metrics"]
    primary = m["primary"]
    spec_out = result["spec"]
    print(f"{result['model']['label']} · {spec_out['task']} · target '{spec_out['target']}'")
    print(f"test {primary}: {m['test'][primary]:.4f}   (baseline {m['baseline'][primary]:.4f})")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    print(f"results written to {out}/")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from forgeml.runner import predict

    rows = predict(Path(args.run_dir), Path(args.data), Path(args.out))
    print(f"{rows} predictions written to {args.out}")
    return 0


def _models(_: argparse.Namespace) -> int:
    for name, entry in MODELS.items():
        params = ", ".join(entry.params.model_fields)
        print(f"{name:32} {entry.task:15} params: {params or '-'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forgeml", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_run = sub.add_parser("run", help="train and evaluate from a spec file")
    p_run.add_argument("spec", help="path to a forgeml.spec/v1 JSON file")
    p_run.add_argument("--out", default="run", help="output directory (default: ./run)")
    p_run.add_argument("--data", help="dataset file, overriding the path in the spec")
    p_run.add_argument("--no-exports", action="store_true", help="skip notebook/script exports")
    p_run.set_defaults(func=_run)
    p_pred = sub.add_parser("predict", help="score new rows with a finished run")
    p_pred.add_argument("run_dir", help="directory produced by `forgeml run`")
    p_pred.add_argument("data", help="CSV, TSV or Parquet file with the same feature columns")
    p_pred.add_argument("--out", default="predictions.csv")
    p_pred.set_defaults(func=_predict)
    sub.add_parser("models", help="list supported models").set_defaults(func=_models)
    sub.add_parser("version", help="print the version").set_defaults(
        func=lambda _: print(__version__) or 0
    )
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ForgeMLError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 2
    except ValidationError as exc:
        print(f"error [invalid_spec]: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
