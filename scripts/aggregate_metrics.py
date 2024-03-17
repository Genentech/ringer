#!/usr/bin/env python

import logging
import pickle
from pathlib import Path
from typing import Any, Union

import typer

from ringer.utils import evaluation


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Union[str, Path], data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def aggregate_metrics(
    mol_metrics_dir: str = "metrics",
    out_dir: str = ".",
) -> None:
    metrics_dir = Path(mol_metrics_dir)
    output_dir = Path(out_dir)
    assert output_dir.exists()

    metrics = {path.name: load_pickle(path) for path in metrics_dir.glob("*.pickle")}

    # Simplify and aggregate results
    metrics = evaluation.CovMatEvaluator.stack_results(metrics)
    metrics_aggregated = evaluation.CovMatEvaluator.aggregate_results(metrics)

    metrics_path = output_dir / "metrics.pickle"
    metrics_aggregated_path = output_dir / "metrics_aggregated.pickle"
    save_pickle(metrics_path, metrics)
    save_pickle(metrics_aggregated_path, metrics_aggregated)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(aggregate_metrics)


if __name__ == "__main__":
    main()
