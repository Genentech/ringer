#!/usr/bin/env python

import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any, Union

import typer
from tqdm.contrib.concurrent import process_map

from ringer.utils import evaluation


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Union[str, Path], data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def compute_metrics(
    mol_dir: str,
    mol_opt_dir: str,
    out_dir: str,
    include_all_atom: bool = True,
    ncpu: int = multiprocessing.cpu_count(),
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    mols_avail = {path.name: load_pickle(path) for path in Path(mol_dir).glob("*.pickle")}
    mols_opt_avail = {path.name: load_pickle(path) for path in Path(mol_opt_dir).glob("*.pickle")}

    # Only keep mols that were generated and reorder so that mols correspond to mols_opt
    mols = {}
    mols_opt = {}
    for name, mol in mols_avail.items():
        try:
            mol_opt = mols_opt_avail[name]
        except KeyError:
            logging.warning(f"Skipping '{name}', no generated mol found")
        else:
            mols[name] = mol
            mols_opt[name] = mol_opt

    # Evaluate
    metric_names = ["ring-rmsd", "ring-tfd"]
    if include_all_atom:
        metric_names.append("rmsd")
    cov_mat_evaluator = evaluation.CovMatEvaluator(metric_names)
    metrics = process_map(cov_mat_evaluator, mols_opt.values(), mols.values(), max_workers=ncpu)
    metrics = dict(zip(mols.keys(), metrics))  # Add names as keys

    # Simplify and aggregate results
    metrics = cov_mat_evaluator.stack_results(metrics)
    metrics_aggregated = cov_mat_evaluator.aggregate_results(metrics)

    metrics_path = output_dir / "metrics.pickle"
    metrics_aggregated_path = output_dir / "metrics_aggregated.pickle"
    save_pickle(metrics_path, metrics)
    save_pickle(metrics_aggregated_path, metrics_aggregated)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(compute_metrics)


if __name__ == "__main__":
    main()
