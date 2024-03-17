from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem
from scipy.optimize import OptimizeResult
from tqdm.contrib.concurrent import process_map

from . import chem, internal_coords


def reconstruct_ring(
    mol: Chem.Mol,
    structure: Dict[str, Any],
    bond_dist_dict: Dict[str, float],
    bond_angle_dict: Optional[Dict[str, float]] = None,
    bond_angle_dev_dict: Optional[Dict[str, float]] = None,
    angles_as_constraints: bool = False,
    opt_init: Literal["best_dists", "average"] = "best_dists",
    skip_opt: bool = False,
    max_conf: Optional[int] = None,
    return_unsuccessful: bool = False,
    ncpu: int = 1,
) -> Union[
    Tuple[Chem.Mol, List[pd.DataFrame]],
    Tuple[Chem.Mol, List[pd.DataFrame], Dict[int, OptimizeResult]],
]:
    """Reconstruct Cartesian coordinates and recover consistent sets of redundant internal
    coordinates.

    Args:
        mol: Molecule containing connectivity and at least one conformer.
        structure: Inconsistent internal coordinates and atom labels for several samples/conformers.
        bond_dist_dict: Bond distances that match the atom labels in structure, usually from the training data.
        bond_angle_dict: If structure doesn't contain bond angles, use these as constraints.
        bond_angle_dev_dict: If structure doesn't contain bond angles, use these as maximum deviations for the bond angle constraints.
        angles_as_constraints: Use the bond angles as constraints instead of targets (default if structure does not contain bond angles).
        opt_init: Initialization method for the optimization.
        skip_opt: Just set the coordinates in sequence and don't run the optimization.
        max_conf: Reconstruct at most this many conformers.
        return_unsuccessful: Return the results objects of unsuccessful optimizations.
        ncpu: Number of processes to use.

    Returns:
        Reconstructed mol with one conformer for each row in the structure dataframes containing
        new Cartesian coordinates of ring atoms and list of reconstructed coordinates for each
        conformer.
    """
    angle_df = structure.get("angle")
    if angle_df is None:
        angles_as_constraints = True
    if angles_as_constraints and bond_angle_dict is None:
        raise ValueError("Must provide bond angles")
    dihedral_df = structure["dihedral"]

    # Set up the optimization class
    ring_idxs = dihedral_df.columns.tolist()
    ring_internal_coords = internal_coords.RingInternalCoordinates(ring_idxs)

    # When reconstructing mol, use mean bond distances from training data
    bond_dists = pd.Series(
        data=(bond_dist_dict[label] for label in structure["atom_labels"]), index=ring_idxs
    )

    bond_angle_devs = None
    if angles_as_constraints:
        bond_angles = pd.Series(
            data=(bond_angle_dict[label] for label in structure["atom_labels"]), index=ring_idxs
        )
        if bond_angle_dev_dict is not None:
            bond_angle_devs = pd.Series(
                data=(bond_angle_dev_dict[label] for label in structure["atom_labels"]),
                index=ring_idxs,
            )

    # Obtain Cartesian coordinates and consistent angles
    # Each row in the dataframes contains the internal coordinates of a sample/conformer
    pfunc = partial(
        _to_cartesian_helper,
        mol=mol,
        ring_internal_coords=ring_internal_coords,
        distance_vals=bond_dists,
        angles_as_constraints=angles_as_constraints,
        angle_vals_max_devs=bond_angle_devs,
        opt_init=opt_init,
        skip_opt=skip_opt,
        print_warning=False,
        return_result_obj=True,
    )
    inputs_list = []
    for conf_idx, dihedrals in dihedral_df.iterrows():
        if not angles_as_constraints:
            bond_angles = angle_df.loc[conf_idx]
        inputs_list.append((bond_angles, dihedrals))
    if max_conf is not None:
        inputs_list = inputs_list[:max_conf]

    chunksize, extra = divmod(len(inputs_list), ncpu * 4)
    if extra:
        chunksize += 1
    results = process_map(pfunc, inputs_list, max_workers=ncpu, chunksize=chunksize)

    coords_opt = [result[0] for result in results]
    unsuccessful_results = {
        conf_idx: result[1]
        for conf_idx, result in zip(dihedral_df.index, results)
        if not result[1].success
    }

    # Make a new mol where the ring atoms contain the new coordinates
    mol_opt = chem.set_atom_positions(mol, coords_opt)

    if return_unsuccessful:
        return mol_opt, coords_opt, unsuccessful_results
    return mol_opt, coords_opt


def _to_cartesian_helper(
    inputs: Tuple[pd.Series, pd.Series],
    ring_internal_coords: internal_coords.RingInternalCoordinates,
    **kwargs: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, OptimizeResult]]:
    """Helper function for to_cartesian."""
    bond_angles, dihedrals = inputs
    return ring_internal_coords.to_cartesian(
        angle_vals_target=bond_angles,
        dihedral_vals_target=dihedrals,
        **kwargs,
    )
