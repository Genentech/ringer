#!/usr/bin/env python

import json
import logging
import multiprocessing
import pickle
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
import ray
import typer
from rdkit import Chem
from tqdm import tqdm

from ringer.utils import reconstruction


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Union[str, Path], data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


@ray.remote
def reconstruct_ring(
    fname: str,
    data: Dict[str, Any],
    bond_dist_dict: Dict[str, float],
    bond_angle_dict: Optional[Dict[str, float]] = None,
    bond_angle_dev_dict: Optional[Dict[str, float]] = None,
    opt_init: Literal["best_dists", "average"] = "best_dists",
    skip_opt: bool = False,
    max_conf: Optional[int] = None,
    return_unsuccessful: bool = False,
    mol_opt_dir: Optional[Union[str, Path]] = None,
) -> Tuple[str, Tuple[Chem.Mol, List[pd.DataFrame]]]:
    mol = data["mol"]
    structure = data["structure"]

    result = reconstruction.reconstruct_ring(
        mol=mol,
        structure=structure,
        bond_dist_dict=bond_dist_dict,
        bond_angle_dict=bond_angle_dict,
        bond_angle_dev_dict=bond_angle_dev_dict,
        opt_init=opt_init,
        skip_opt=skip_opt,
        max_conf=max_conf,
        return_unsuccessful=return_unsuccessful,
    )

    if mol_opt_dir is not None:
        mol_opt = result[0]
        mol_opt_path = Path(mol_opt_dir) / Path(fname).name
        save_pickle(mol_opt_path, mol_opt)

    return fname, result


def get_as_iterator(obj_ids):
    # Returns results as they're ready
    # Order is preserved within the IDs that are ready,
    # but not if this iterator is converted to a list
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def reconstruct(
    mol_dir: str,
    structures_path: str,
    out_dir: str,
    mean_distances_path: str,
    mean_angles_path: Optional[str] = None,
    std_angles_path: Optional[str] = None,
    opt_init: str = "best_dists",
    skip_opt: bool = False,
    max_conf: Optional[int] = None,
    save_unsuccessful: bool = False,
    ncpu: int = multiprocessing.cpu_count(),
) -> None:
    output_dir = Path(out_dir)
    output_dir.mkdir(exist_ok=True)

    # Load data
    mols = {path.name: load_pickle(path) for path in Path(mol_dir).glob("*.pickle")}
    structures_dict = load_pickle(structures_path)
    mean_bond_distances = load_json(mean_distances_path)
    mean_bond_angles = None if mean_angles_path is None else load_json(mean_angles_path)
    std_bond_angles = None if std_angles_path is None else load_json(std_angles_path)

    # Get mols in correct order
    mols_and_structures = {
        fname: dict(mol=mols[Path(fname).name], structure=structure)
        for fname, structure in structures_dict.items()
    }

    mol_opt_dir = output_dir / "reconstructed_mols"
    mol_opt_dir.mkdir(exist_ok=True)

    # Reconstruct
    if skip_opt:
        logging.info("Skipping opt")
    ray.init(num_cpus=ncpu)
    result_ids = [
        reconstruct_ring.remote(
            fname,
            mol_and_structure,
            mean_bond_distances,
            bond_angle_dict=mean_bond_angles,
            bond_angle_dev_dict=std_bond_angles,
            opt_init=opt_init,
            skip_opt=skip_opt,
            max_conf=max_conf,
            return_unsuccessful=save_unsuccessful,
            mol_opt_dir=mol_opt_dir,
        )
        for fname, mol_and_structure in mols_and_structures.items()
    ]
    mols_and_coords_opt = dict(tqdm(get_as_iterator(result_ids), total=len(result_ids)))

    # Post-process and dump data
    def get_structure_from_coords(coords: List[pd.DataFrame], name: str) -> Dict[str, Any]:
        # Convert list of coords to structure
        # Concatenate making hierarchical index of sample_idx and atom_idx
        coords_stacked = pd.concat(
            coords, keys=range(len(coords)), names=["sample_idx", "atom_idx"]
        )
        # Pivot so we can get all samples for each feature from the outermost column
        coords_pivoted = coords_stacked.unstack(level="atom_idx")
        structure = {
            feat_name: coords_pivoted[feat_name] for feat_name in coords_pivoted.columns.levels[0]
        }
        structure["atom_labels"] = structures_dict[name]["atom_labels"]
        return structure

    reconstructed_structures_dict = {}
    unsuccessful_results = {}
    for fname, result in mols_and_coords_opt.items():
        coords_opt = result[1]

        structure = get_structure_from_coords(coords_opt, fname)
        reconstructed_structures_dict[fname] = structure

        if save_unsuccessful:
            result_objs = result[2]
            if result_objs:
                unsuccessful_results[fname] = result_objs

    save_pickle(output_dir / "samples_reconstructed.pickle", reconstructed_structures_dict)
    if unsuccessful_results:
        save_pickle(output_dir / "unsuccessful_opts.pickle", unsuccessful_results)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(reconstruct)


if __name__ == "__main__":
    main()
