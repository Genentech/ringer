#!/usr/bin/env python

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import typer
from rdkit import Chem

import ringer
from ringer.sidechain_reconstruction import (
    Macrocycle,
    Reconstructor,
    set_rdkit_geometries,
)
from ringer.utils import reconstruction

RECONSTRUCTION_DATA_PATH = (
    Path(ringer.__file__).resolve().parent
    / "sidechain_reconstruction/data/reconstruction_data.pickle"
)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Union[str, Path], data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def merge_samples(
    sample_backbone: Dict[str, Any], sample_sidechain: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge samples from two sources corresponding to reconstructed backbones and the original
    sidechain predictions that do not get modified during reconstruction."""
    merged_dict = {}
    merged_dict["angle"] = sample_backbone["angle"]
    merged_dict["dihedral"] = sample_backbone["dihedral"]

    sidechain_internals = [
        "sc_a0",
        "sc_a1",
        "sc_a2",
        "sc_a3",
        "sc_a4",
        "sc_chi0",
        "sc_chi1",
        "sc_chi2",
        "sc_chi3",
        "sc_chi4",
    ]

    for sc_key in sidechain_internals:
        merged_dict[sc_key] = sample_sidechain[sc_key]
    return merged_dict


def reconstruct(
    idx: int,
    mol_dir: str,
    structures_path: str,
    out_dir: str,
    mean_distances_path: str,
    reconstruct_sidechains: bool = True,
    mean_angles_path: Optional[str] = None,
    std_angles_path: Optional[str] = None,
    angles_as_constraints: bool = False,
    opt_init: str = "best_dists",
    skip_opt: bool = False,
    max_conf: Optional[int] = None,
    ncpu: int = 1,
) -> None:
    mol_opt_dir = Path(out_dir)
    mol_opt_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    structures_dict = load_pickle(structures_path)
    mean_bond_distances = load_json(mean_distances_path)
    mean_bond_angles = None if mean_angles_path is None else load_json(mean_angles_path)
    std_bond_angles = None if std_angles_path is None else load_json(std_angles_path)

    # Load mol
    fname = list(structures_dict.keys())[idx]
    mol_name = Path(fname).name
    mol_path = Path(mol_dir) / mol_name
    mol = load_pickle(mol_path)
    structure = structures_dict[fname]

    # Reconstruct
    logging.info(f"Reconstructing {mol_name}")
    if skip_opt:
        logging.info("Skipping opt")

    start_time = time.time()
    result = reconstruction.reconstruct_ring(
        mol=mol,
        structure=structure,
        bond_dist_dict=mean_bond_distances,
        bond_angle_dict=mean_bond_angles,
        bond_angle_dev_dict=std_bond_angles,
        angles_as_constraints=angles_as_constraints,
        opt_init=opt_init,
        skip_opt=skip_opt,
        max_conf=max_conf,
        return_unsuccessful=False,  # Don't save unsuccessful optimizations for now
        ncpu=ncpu,
    )
    end_time = time.time()
    logging.info(f"Reconstruction took {end_time - start_time:.2f} seconds")

    mol_opt = result[0]

    # Post-process and dump data
    def get_structure_from_coords(coords: List[pd.DataFrame]) -> Dict[str, Any]:
        # Convert list of coords to structure
        # Concatenate making hierarchical index of sample_idx and atom_idx
        coords_stacked = pd.concat(
            coords, keys=range(len(coords)), names=["sample_idx", "atom_idx"]
        )
        # Pivot so we can get all samples for each feature from the outermost column
        coords_pivoted = coords_stacked.unstack(level="atom_idx")
        new_structure = {
            feat_name: coords_pivoted[feat_name] for feat_name in coords_pivoted.columns.levels[0]
        }
        new_structure["atom_labels"] = structure["atom_labels"]
        return new_structure

    if reconstruct_sidechains:
        logging.info("Reconstructing sidechains")

        with open(RECONSTRUCTION_DATA_PATH, "rb") as f:
            reconstruction_config = pickle.load(f)

        coords_opt = result[1]
        structure_opt = get_structure_from_coords(coords_opt)

        # Merge with original sidechain predictions
        sample = merge_samples(structure_opt, structure)

        mol_opt_no_h = Chem.RemoveHs(mol_opt)
        mc = Macrocycle(
            mol_opt_no_h,
            reconstruction_config,
            coords=False,
            copy=True,
            verify=True,
        )
        reconstructor = Reconstructor(mc)

        internals_tensor = reconstructor.parse_internals(sample)
        index_tensor = reconstructor.stacked_tuples

        positions = reconstructor.reconstruct(internals_tensor, index_tensor)
        mol_opt = set_rdkit_geometries(mol_opt_no_h, positions, copy=True)

    mol_opt_path = mol_opt_dir / mol_name
    save_pickle(mol_opt_path, mol_opt)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(reconstruct)


if __name__ == "__main__":
    main()
