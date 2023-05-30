from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from rdkit import Chem
from scipy.optimize import OptimizeResult

from . import chem, internal_coords


def reconstruct_ring(
    mol: Chem.Mol,
    structure: Dict[str, Any],
    bond_dist_dict: Dict[str, float],
    bond_angle_dict: Optional[Dict[str, float]] = None,
    bond_angle_dev_dict: Optional[Dict[str, float]] = None,
    skip_opt: bool = False,
    return_unsuccessful: bool = False,
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
        skip_opt: Just set the coordinates in sequence and don't run the optimization.
        return_unsuccessful: Return the results objects of unsuccessful optimizations.

    Returns:
        Reconstructed mol with one conformer for each row in the structure dataframes containing
        new Cartesian coordinates of ring atoms and list of reconstructed coordinates for each
        conformer.
    """
    # Recover consistent set of redundant internal coordinates given a sample from a conditional model and convert to Cartesian coordinates
    try:
        angle_df = structure["angle"]
    except KeyError:
        if bond_angle_dict is None:
            raise ValueError("Must provide bond angles")
        else:
            angle_df = None
    dihedral_df = structure["dihedral"]

    # Set up the optimization class
    ring_idxs = dihedral_df.columns.tolist()
    ring_internal_coords = internal_coords.RingInternalCoordinates(ring_idxs)

    # When reconstructing mol, use mean bond distances from training data
    bond_dists = pd.Series(
        data=(bond_dist_dict[label] for label in structure["atom_labels"]), index=ring_idxs
    )

    bond_angle_devs = None
    if angle_df is None:
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
    coords_opt = []
    unsuccessful_results = {}
    for conf_idx, dihedrals in dihedral_df.iterrows():
        if angle_df is not None:
            bond_angles = angle_df.loc[conf_idx]
        coords_df, result = ring_internal_coords.to_cartesian(
            mol,
            distance_vals=bond_dists,
            angle_vals_target=bond_angles,
            dihedral_vals_target=dihedrals,
            angles_as_constraints=angle_df is None,
            angle_vals_max_devs=bond_angle_devs,
            skip_opt=skip_opt,
            print_warning=False,
            return_result_obj=True,
        )
        coords_opt.append(coords_df)
        if not result.success:
            unsuccessful_results[conf_idx] = result

    # Make a new mol where the ring atoms contain the new coordinates
    mol_opt = chem.set_atom_positions(mol, coords_opt)

    if return_unsuccessful:
        return mol_opt, coords_opt, unsuccessful_results
    return mol_opt, coords_opt
