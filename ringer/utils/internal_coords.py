"""Code to compute internal coordinates and their derivatives."""
import logging
import pickle
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from geometric.internal import Angle, Dihedral, Distance
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import (
    LinearConstraint,
    NonlinearConstraint,
    OptimizeResult,
    minimize,
)

from . import chem, utils

BACKBONE_ATOM_LABELS = ["N", "Calpha", "CO"]
BACKBONE_ATOM_IDS = [0, 1, 2]
BACKBONE_ATOM_ID_TO_LABEL = dict(zip(BACKBONE_ATOM_IDS, BACKBONE_ATOM_LABELS))


def get_macrocycle_distances_and_angles_from_file(
    fname: Union[str, Path]
) -> Dict[str, Union[List[str], List[int], pd.DataFrame]]:
    with open(fname, "rb") as f:
        ensemble_data = pickle.load(f)
    mol = ensemble_data["rd_mol"]

    # Get macrocycle indices in N to C direction starting at an N
    macrocycle_idxs = chem.get_macrocycle_idxs(mol, n_to_c=True)
    if macrocycle_idxs is None:
        raise ValueError(f"Could not get macrocycle indices for '{fname}'")
    assert len(macrocycle_idxs) % 3 == 0
    reps = len(macrocycle_idxs) // 3
    atom_labels = BACKBONE_ATOM_LABELS * reps
    atom_ids = BACKBONE_ATOM_IDS * reps

    distances = get_macrocycle_bond_distances(mol, macrocycle_idxs)
    angles = get_macrocycle_bond_angles(mol, macrocycle_idxs)
    dihedrals = get_macrocycle_dihedrals(mol, macrocycle_idxs)

    return {
        "atom_labels": atom_labels,
        "atom_ids": atom_ids,
        "distance": distances,
        "angle": angles,
        "dihedral": dihedrals,
    }


def get_macrocycle_bond_idxs(macrocycle_idxs: List[int]) -> List[Tuple[int, int]]:
    return list(utils.get_overlapping_sublists(macrocycle_idxs, 2))


def get_macrocycle_angle_idxs(macrocycle_idxs: List[int]) -> List[Tuple[int, int, int]]:
    # Shift indices by one
    macrocycle_idxs = macrocycle_idxs.copy()
    macrocycle_idxs.insert(0, macrocycle_idxs.pop())
    return list(utils.get_overlapping_sublists(macrocycle_idxs, 3))


def get_macrocycle_dihedral_idxs(
    macrocycle_idxs: List[int],
) -> List[Tuple[int, int, int, int]]:
    # For the first dihedral to correspond to the bond between the first two atoms, the
    # first element in the list has to be the last atom
    macrocycle_idxs = macrocycle_idxs.copy()
    macrocycle_idxs.insert(0, macrocycle_idxs.pop())
    return list(utils.get_overlapping_sublists(macrocycle_idxs, 4))


def get_macrocycle_bond_distances(mol: Chem.Mol, macrocycle_idxs: List[int]) -> pd.DataFrame:
    bond_idxs = get_macrocycle_bond_idxs(macrocycle_idxs)
    macrocycle_distances = defaultdict(list)

    # Dictionary keys are the first atoms of each bond
    for conformer in mol.GetConformers():
        for atom_idx, bond_atom_idxs in zip(macrocycle_idxs, bond_idxs):
            macrocycle_distances[atom_idx].append(
                AllChem.GetBondLength(conformer, *bond_atom_idxs)
            )

    # Each row in dataframe are the distances for a conformer
    distance_df = pd.DataFrame(data=macrocycle_distances)
    distance_df.index.name = "conf_idx"

    return distance_df


def get_macrocycle_bond_angles(mol: Chem.Mol, macrocycle_idxs: List[int]) -> pd.DataFrame:
    angle_idxs = get_macrocycle_angle_idxs(macrocycle_idxs)
    macrocycle_angles = defaultdict(list)

    # Dictionary keys are the central atoms of each angle
    for conformer in mol.GetConformers():
        for atom_idx, angle_atom_idxs in zip(macrocycle_idxs, angle_idxs):
            macrocycle_angles[atom_idx].append(AllChem.GetAngleRad(conformer, *angle_atom_idxs))

    # Each row in dataframe are the angles for a conformer
    angle_df = pd.DataFrame(data=macrocycle_angles)
    angle_df.index.name = "conf_idx"

    return angle_df


def get_macrocycle_dihedrals(mol: Chem.Mol, macrocycle_idxs: List[int]) -> pd.DataFrame:
    dihedral_idxs = get_macrocycle_dihedral_idxs(macrocycle_idxs)
    macrocycle_dihedrals = defaultdict(list)

    # Dictionary keys are the first atoms of each bond
    for conformer in mol.GetConformers():
        for atom_idx, dihedral_atom_idxs in zip(macrocycle_idxs, dihedral_idxs):
            macrocycle_dihedrals[atom_idx].append(
                AllChem.GetDihedralRad(conformer, *dihedral_atom_idxs)
            )

    # Each row in dataframe are the dihedrals for a conformer
    dihedral_df = pd.DataFrame(data=macrocycle_dihedrals)
    dihedral_df.index.name = "conf_idx"

    return dihedral_df


def modify_macrocycle_geometry(
    mol: Chem.Mol,  # Should have one conformer with 3D geometry
    bond_dists: Union[np.ndarray, pd.Series],
    bond_angles: Union[np.ndarray, pd.Series],
    dihedrals: Union[np.ndarray, pd.Series],
    macrocycle_idxs: Optional[List[int]] = None,
    shift: int = 0,  # Shift indices to the right by this amount
) -> Chem.Mol:
    """Sequentially set distances, angles, and dihedrals of macrocycle.

    Extraneous degrees of freedom are skipped, i.e., last distance, first and last angle, first and
    last two dihedrals.
    """
    assert len(bond_dists) == len(bond_angles) == len(dihedrals)
    if macrocycle_idxs is None:
        assert isinstance(bond_dists, pd.Series)
        macrocycle_idxs = bond_dists.index
        bond_dists = bond_dists.to_numpy()
    else:
        assert len(macrocycle_idxs) == len(bond_dists)

    if isinstance(bond_angles, pd.Series):
        bond_angles = bond_angles.loc[macrocycle_idxs].to_numpy()
    if isinstance(dihedrals, pd.Series):
        dihedrals = dihedrals.loc[macrocycle_idxs].to_numpy()

    macrocycle_idxs = list(map(int, macrocycle_idxs))

    mol = Chem.Mol(mol)  # Make copy

    # Circular shift (also works for negative shift)
    macrocycle_idxs = macrocycle_idxs[-shift:] + macrocycle_idxs[:-shift]
    bond_dists = np.roll(bond_dists, shift)
    bond_angles = np.roll(bond_angles, shift)
    dihedrals = np.roll(dihedrals, shift)

    rwmol = Chem.RWMol(mol)

    # Temporarily remove all bonds so that we can set all internal coordinates without
    # running into issues due to atoms being in a ring (e.g., when there's a proline)
    removed_bonds = {}
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        bt = bond.GetBondType()
        rwmol.RemoveBond(idx1, idx2)
        removed_bonds[(idx1, idx2)] = bt

    # Add back in all but the last ring bond so that we can set distances and angles in
    # the macrocycle sequentially
    bond_idxs = get_macrocycle_bond_idxs(macrocycle_idxs)
    for idx1, idx2 in bond_idxs[:-1]:
        key = (idx1, idx2)
        key = key if key in removed_bonds else (idx2, idx1)
        bt = removed_bonds[key]
        rwmol.AddBond(idx1, idx2, bt)
        del removed_bonds[key]

    # Initialize ring info so that we don't get errors when setting internal coords
    Chem.GetSSSR(rwmol)

    conf = rwmol.GetConformer()
    angle_idxs = get_macrocycle_angle_idxs(macrocycle_idxs)
    dihedral_idxs = get_macrocycle_dihedral_idxs(macrocycle_idxs)

    # Skip bond that we removed
    for i in range(len(bond_idxs) - 1):
        idxs = bond_idxs[i]
        val = float(bond_dists[i])
        AllChem.SetBondLength(conf, *idxs, val)

    # Skip first and last bond angle because we removed bond
    for i in range(1, len(angle_idxs) - 1):
        idxs = angle_idxs[i]
        val = float(bond_angles[i])
        AllChem.SetAngleRad(conf, *idxs, val)

    # Skip first and last two dihedrals because we removed bond
    for i in range(1, len(dihedral_idxs) - 2):
        idxs = dihedral_idxs[i]
        val = float(dihedrals[i])
        AllChem.SetDihedralRad(conf, *idxs, val)

    # Add back in all the other bonds
    for (idx1, idx2), bt in removed_bonds.items():
        rwmol.AddBond(idx1, idx2, bt)

    new_mol = rwmol.GetMol()
    return new_mol


def set_macrocycle_geometry_with_best_dists(
    mol: Chem.Mol,
    bond_dists: Union[np.ndarray, pd.Series],
    bond_angles: Union[np.ndarray, pd.Series],
    dihedrals: Union[np.ndarray, pd.Series],
    macrocycle_idxs: Optional[List[int]] = None,
) -> Chem.Mol:
    """Sequentially set distances, angles, and dihedrals of macrocycle starting at each atom and
    return the molecule that has the smallest distance error."""
    min_dist_sse = float("inf")
    mol_best_dists = None

    if macrocycle_idxs is None:
        assert isinstance(bond_dists, pd.Series)
        macrocycle_idxs = bond_dists.index.tolist()

    for shift in range(len(macrocycle_idxs)):
        mol_tmp = modify_macrocycle_geometry(
            mol,
            bond_dists,
            bond_angles,
            dihedrals,
            macrocycle_idxs=macrocycle_idxs,
            shift=shift,
        )

        actual_bond_dists = get_macrocycle_bond_distances(mol_tmp, macrocycle_idxs).loc[0]
        distance_sse = np.sum((actual_bond_dists - bond_dists) ** 2)

        if distance_sse < min_dist_sse:
            min_dist_sse = distance_sse
            mol_best_dists = mol_tmp

    return mol_best_dists


class InternalCoordinates:
    def __init__(
        self,
        bond_idxs: Optional[List[Tuple[int, int]]] = None,
        angle_idxs: Optional[List[Tuple[int, int, int]]] = None,
        dihedral_idxs: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> None:
        if bond_idxs is None:
            bond_idxs = []
        if angle_idxs is None:
            angle_idxs = []
        if dihedral_idxs is None:
            dihedral_idxs = []

        self.distances = [Distance(*idxs) for idxs in bond_idxs]
        self.angles = [Angle(*idxs) for idxs in angle_idxs]
        self.dihedrals = [Dihedral(*idxs) for idxs in dihedral_idxs]

    def compute_distances(self, xyz: np.ndarray) -> np.ndarray:
        return np.array([distance.value(xyz) for distance in self.distances])

    def compute_angles(self, xyz: np.ndarray) -> np.ndarray:
        return np.array([angle.value(xyz) for angle in self.angles])

    def compute_dihedrals(self, xyz: np.ndarray) -> np.ndarray:
        return np.array([dihedral.value(xyz) for dihedral in self.dihedrals])

    def compute_distance_jacobian(self, xyz: np.ndarray) -> np.ndarray:
        return np.array([distance.derivative(xyz).ravel() for distance in self.distances])

    def compute_angle_jacobian(self, xyz: np.ndarray) -> np.ndarray:
        return np.array([angle.derivative(xyz).ravel() for angle in self.angles])

    def compute_dihedral_jacobian(self, xyz: np.ndarray) -> np.ndarray:
        return np.array([dihedral.derivative(xyz).ravel() for dihedral in self.dihedrals])


class RingInternalCoordinates(InternalCoordinates):
    """Create internal coordinate object for given ring indices containing all bonds, angles, and
    dihedrals in the ring."""

    def __init__(self, ring_idxs: List[int]) -> None:
        # The bond, angle, and dihedral indices are NOT numbered based on ring_idxs,
        # instead they correspond to generic indices for a monocyclic ring (i.e.,
        # indices from 0 to number of atoms in the ring minus one). ring_idxs is only
        # needed in order to extract the correct Cartesian coordinates when provided
        # with a Chem.Mol object
        atom_idxs = list(range(len(ring_idxs)))
        bond_idxs = get_macrocycle_bond_idxs(atom_idxs)  # Pairs
        angle_idxs = get_macrocycle_angle_idxs(atom_idxs)  # Triples
        dihedral_idxs = get_macrocycle_dihedral_idxs(atom_idxs)  # Quartets
        super().__init__(bond_idxs, angle_idxs, dihedral_idxs)

        self.ring_idxs = ring_idxs

    def to_cartesian(
        self,
        mol: Chem.Mol,
        distance_vals: Union[np.ndarray, pd.Series],
        angle_vals_target: Union[np.ndarray, pd.Series],
        dihedral_vals_target: Union[np.ndarray, pd.Series],
        angles_as_constraints: bool = False,
        angle_vals_max_devs: Optional[Union[np.ndarray, pd.Series]] = None,
        use_orientation_constraints: bool = False,
        skip_opt: bool = False,  # Just return the results from setting coords in sequence
        print_warning: bool = True,
        return_result_obj: bool = False,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, OptimizeResult]]:
        """Get the Cartesian coordinates of ring atoms in a molecule given a set of redundant (and
        possibly inconsistent) internal bond angles and dihedrals such that the bond distances are
        exactly satisfied."""
        if isinstance(distance_vals, pd.Series):
            distance_vals = distance_vals.loc[self.ring_idxs]
        if isinstance(angle_vals_target, pd.Series):
            angle_vals_target = angle_vals_target.loc[self.ring_idxs]
        if isinstance(dihedral_vals_target, pd.Series):
            dihedral_vals_target = dihedral_vals_target.loc[self.ring_idxs]
        if angle_vals_max_devs is not None:
            if isinstance(angle_vals_max_devs, pd.Series):
                angle_vals_max_devs = angle_vals_max_devs.loc[self.ring_idxs]

        # Get initial guess using sequential approach to setting angles/dihedrals, and
        # select the molecule with the distances that most closely match the true ones
        mol0 = set_macrocycle_geometry_with_best_dists(
            mol,
            distance_vals,
            angle_vals_target,
            dihedral_vals_target,
            macrocycle_idxs=self.ring_idxs,
        )

        xyz0 = mol0.GetConformer().GetPositions()[self.ring_idxs]

        if skip_opt:
            xyz_opt = xyz0

            class DummyResult:
                success = True

            result = DummyResult
        else:
            # Transform to canonical orientation and position
            trans_mat = compute_canonical_transform(xyz0)
            xyz0 = apply_affine_transform(xyz0, trans_mat)
            xyz0 = xyz0.ravel()

            if angles_as_constraints:
                obj_func = partial(
                    dihedral_loss,
                    internal_coords=self,
                    dihedral_vals_target=np.asarray(dihedral_vals_target),
                    grad=True,
                )
            else:
                obj_func = partial(
                    angle_and_dihedral_loss,
                    internal_coords=self,
                    angle_vals_target=np.asarray(angle_vals_target),
                    dihedral_vals_target=np.asarray(dihedral_vals_target),
                    grad=True,
                )

            # Equality constraint for distances
            distance_constraint = NonlinearConstraint(
                self.compute_distances,
                lb=np.asarray(distance_vals),
                ub=np.asarray(distance_vals),
                jac=self.compute_distance_jacobian,
            )
            constraints = [distance_constraint]

            if angles_as_constraints:
                angle_constraint = NonlinearConstraint(
                    self.compute_angles,
                    lb=np.asarray(angle_vals_target) - np.asarray(angle_vals_max_devs),
                    ub=np.asarray(angle_vals_target) + np.asarray(angle_vals_max_devs),
                    jac=self.compute_angle_jacobian,
                )
                constraints.append(angle_constraint)

            if use_orientation_constraints:
                # Constrain centroid to coincide with origin
                # Jacobian of centroid is the linear constraint matrix
                centroid_constraint = LinearConstraint(
                    compute_centroid_jacobian(xyz0),  # xyz0 only needed for matrix dimensions
                    lb=np.zeros(3),
                    ub=np.zeros(3),
                )
                # Constrain off-diagonal elements of gyration tensor to be zero, which means
                # that the principal axes of rotation are aligned with the x-, y-, and z-axes
                # Only need to constrain upper triangular part due to symmetry
                rotation_constraint = NonlinearConstraint(
                    lambda xyz: compute_gyration_tensor(xyz)[[0, 0, 1], [1, 2, 2]],
                    lb=np.zeros(3),
                    ub=np.zeros(3),
                    jac=lambda xyz: compute_gyration_tensor_derivative(xyz)[[0, 0, 1], [1, 2, 2]],
                )
                constraints.extend(
                    [
                        centroid_constraint,
                        rotation_constraint,
                    ]
                )

            result = minimize(
                obj_func,
                xyz0,
                constraints=constraints,
                jac=True,
                options=dict(maxiter=100),
            )
            if not result.success and print_warning:
                logging.warning(
                    f"Optimization terminated unsuccessfully with message: {result.message}"
                )
            xyz_opt = result.x

        angle_vals_opt = self.compute_angles(xyz_opt)
        dihedral_vals_opt = self.compute_dihedrals(xyz_opt)
        internal_coords_opt = pd.DataFrame(
            data={"angle": angle_vals_opt, "dihedral": dihedral_vals_opt}, index=self.ring_idxs
        )

        coords_opt = pd.DataFrame(
            data=xyz_opt.reshape(-1, 3), index=self.ring_idxs, columns=["x", "y", "z"]
        )
        coords_opt = pd.concat([coords_opt, internal_coords_opt], axis=1)

        if return_result_obj:
            return coords_opt, result
        return coords_opt


def compute_centroid(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.reshape(-1, 3)
    return np.mean(xyz, axis=0)


def compute_centroid_jacobian(xyz: np.ndarray) -> np.ndarray:
    # Jacobian is constant, xyz only needed for its size
    xyz = xyz.reshape(-1, 3)
    num_atoms = len(xyz)
    sum_jac_x = np.tile([1.0, 0.0, 0.0], num_atoms)
    sum_jac_y = np.tile([0.0, 1.0, 0.0], num_atoms)
    sum_jac_z = np.tile([0.0, 0.0, 1.0], num_atoms)
    return np.vstack([sum_jac_x, sum_jac_y, sum_jac_z]) / num_atoms


def compute_gyration_tensor(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz.reshape(-1, 3)
    xyz = xyz - compute_centroid(xyz)
    return xyz.T @ xyz / len(xyz)


def compute_gyration_tensor_derivative(xyz: np.ndarray) -> np.ndarray:
    """Define this derivative as the 3D tensor where the derivative of element (i,j) of the
    gyration tensor is computed as a vector that is extended into the third dimension."""
    xyz = xyz.reshape(-1, 3)
    xyz = xyz - compute_centroid(xyz)
    x, y, z = xyz.T

    num_atoms = len(xyz)
    der = np.zeros((3, 3, num_atoms * 3))

    # Diagonal
    der[0, 0, ::3] = 2 * x
    der[1, 1, 1::3] = 2 * y
    der[2, 2, 2::3] = 2 * z

    # Derivative of xy & yx
    der[0, 1, ::3] = y
    der[0, 1, 1::3] = x
    der[1, 0] = der[0, 1]

    # Derivative of xz & zx
    der[0, 2, ::3] = z
    der[0, 2, 2::3] = x
    der[2, 0] = der[0, 2]

    # Derivative of yz & zy
    der[1, 2, 1::3] = z
    der[1, 2, 2::3] = y
    der[2, 1] = der[1, 2]

    return der / num_atoms


def compute_canonical_transform(xyz: np.ndarray) -> np.ndarray:
    """Compute 4x4 transformation matrix using homoegenous coordinates representing rotation and
    translation of the given Cartesian coordinates such that the principal axes correspond to the
    xyz-axes and the centroid is at the origin."""
    centroid = compute_centroid(xyz)
    gyration_tensor = compute_gyration_tensor(xyz)

    _, eig_vecs = np.linalg.eigh(gyration_tensor)
    eig_vecs = np.fliplr(eig_vecs)  # In order of largest to smallest eig val
    signs = np.where(eig_vecs.sum(axis=0) > 0, 1.0, -1.0)  # Choose appropriate signs
    rot_mat = (signs * eig_vecs).T

    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    trans_mat[:3, 3] = -rot_mat @ centroid

    return trans_mat


def apply_affine_transform(xyz: np.ndarray, trans_mat: np.ndarray) -> np.ndarray:
    """Apply 4x4 transformation matrix to Cartesian coordinates."""
    xyz = xyz.reshape(-1, 3)

    # Get homogeneous coordinates
    xyz_hom = np.ones((len(xyz), 4))
    xyz_hom[:, :3] = xyz

    return (trans_mat @ xyz_hom.T).T[:, :3]


def angle_loss(
    xyz: np.ndarray,  # 1D array of flattened Cartesian coords
    internal_coords: InternalCoordinates,
    angle_vals_target: np.ndarray,
    grad: bool = False,  # Also compute and return analytical gradient
) -> Union[float, Tuple[float, float]]:
    """Compute squared error loss between target angles and angles computed from the given
    Cartesian coordinates.

    internal_coords must contain the appropriate angles that correspond to angle_vals_target when
    evaluated.
    """
    angle_vals = internal_coords.compute_angles(xyz)
    angle_diffs = angle_vals - angle_vals_target
    loss = np.sum(angle_diffs**2)

    if grad:
        angle_jac = internal_coords.compute_angle_jacobian(xyz)
        gradient = 2 * np.dot(angle_diffs, angle_jac)

        return loss, gradient

    return loss


def dihedral_loss(
    xyz: np.ndarray,  # 1D array of flattened Cartesian coords
    internal_coords: InternalCoordinates,
    dihedral_vals_target: np.ndarray,
    grad: bool = False,  # Also compute and return analytical gradient
) -> Union[float, Tuple[float, float]]:
    """Compute squared error loss between target dihedrals and dihedrals computed from the given
    Cartesian coordinates.

    internal_coords must contain the appropriate dihedrals that correspond to dihedral_vals_target
    when evaluated.
    """
    dihedral_vals = internal_coords.compute_dihedrals(xyz)
    dihedral_diffs = utils.modulo_with_wrapped_range(dihedral_vals - dihedral_vals_target)
    loss = np.sum(dihedral_diffs**2)

    if grad:
        dihedral_jac = internal_coords.compute_dihedral_jacobian(xyz)
        gradient = 2 * np.dot(dihedral_diffs, dihedral_jac)

        return loss, gradient

    return loss


def angle_and_dihedral_loss(
    xyz: np.ndarray,  # 1D array of flattened Cartesian coords
    internal_coords: InternalCoordinates,
    angle_vals_target: np.ndarray,
    dihedral_vals_target: np.ndarray,
    dihedral_weight: float = 1.0,  # Weight of dihedral loss relative to angle loss
    grad: bool = False,  # Also compute and return analytical gradient
) -> Union[float, Tuple[float, float]]:
    """Compute squared error loss between target angles/dihedrals and angles/dihedrals computed
    from the given Cartesian coordinates.

    internal_coords must contain the appropriate angles and dihedrals that correspond to
    angle_vals_target and dihedral_vals_target when evaluated.
    """
    angle_loss_val = angle_loss(
        xyz=xyz,
        internal_coords=internal_coords,
        angle_vals_target=angle_vals_target,
        grad=grad,
    )
    dihedral_loss_val = dihedral_loss(
        xyz=xyz,
        internal_coords=internal_coords,
        dihedral_vals_target=dihedral_vals_target,
        grad=grad,
    )

    if grad:
        angle_loss_val, angle_grad_val = angle_loss_val
        dihedral_loss_val, dihedral_grad_val = dihedral_loss_val
        gradient = angle_grad_val + dihedral_weight * dihedral_grad_val

    loss = angle_loss_val + dihedral_weight * dihedral_loss_val

    if grad:
        return loss, gradient
    return loss
