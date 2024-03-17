#!/usr/bin/env python

import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Union

import typer
from rdkit import Chem

from ringer.utils import evaluation


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Union[str, Path], data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def remove_confs(mol: Chem.Mol, confs_to_keep: int = 1) -> Chem.Mol:
    new_mol = Chem.Mol(mol)
    conf_ids = [conf.GetId() for conf in new_mol.GetConformers()]
    conf_ids_to_remove = conf_ids[confs_to_keep:]
    for conf_id in conf_ids_to_remove:
        new_mol.RemoveConformer(conf_id)
    return new_mol


def compute_metrics(
    mol_true_path: str,
    mol_reconstructed_path: str,
    max_true_confs: Optional[int] = None,
    out_dir: str = "metrics",
    include_all_atom: bool = True,
    ncpu: int = 1,
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mol_path = Path(mol_true_path)
    mol_opt_path = Path(mol_reconstructed_path)
    assert mol_path.name == mol_opt_path.name

    if not mol_opt_path.exists():
        raise IOError(f"'{mol_opt_path}' is missing")

    # Load data
    mol = load_pickle(mol_path)
    mol_opt = load_pickle(mol_opt_path)

    if max_true_confs is not None:
        mol = remove_confs(mol, max_true_confs)

    # Evaluate
    metric_names = ["ring-rmsd", "ring-tfd"]
    if include_all_atom:
        metric_names.append("rmsd")
    cov_mat_evaluator = evaluation.CovMatEvaluator(metric_names)
    metrics = cov_mat_evaluator(mol_opt, mol, ncpu=ncpu)

    metrics_path = output_dir / mol_path.name
    save_pickle(metrics_path, metrics)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(compute_metrics)


if __name__ == "__main__":
    main()
