from typing import List, Optional, Set, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from . import utils

# Carboxyl before nitrogen corresponds to N-to-C direction of peptide
PEPTIDE_PATTERN = Chem.MolFromSmarts("[OX1]=[C;R][N;R]")


def get_macrocycle_idxs(
    mol: Chem.Mol, min_size: int = 9, n_to_c: bool = True
) -> Optional[List[int]]:
    sssr = Chem.GetSymmSSSR(mol)
    if len(sssr) > 0:
        largest_ring = max(sssr, key=len)
        if len(largest_ring) >= min_size:
            idxs = list(largest_ring)
            if n_to_c:
                return macrocycle_idxs_in_n_to_c_direction(mol, idxs)
            return idxs
    return None


def macrocycle_idxs_in_n_to_c_direction(mol: Chem.Mol, macrocycle_idxs: List[int]) -> List[int]:
    # Obtain carbon and nitrogen idxs in peptide bonds in the molecule
    matches = mol.GetSubstructMatches(PEPTIDE_PATTERN)
    if not matches:
        raise ValueError("Did not match any peptide bonds")

    # We match 3 atoms each time (O, C, N), just need C and N in the ring
    carbon_and_nitrogen_idxs = {match[1:] for match in matches}

    for atom_idx_pair in utils.get_overlapping_sublists(macrocycle_idxs, 2):
        # If the directionality of atom idxs is already in N to C direction, then pairs of these
        # atom indices should already be in the set of matched atoms, otherwise, we need to flip
        # the direction
        if tuple(atom_idx_pair) in carbon_and_nitrogen_idxs:
            break
    else:
        macrocycle_idxs = macrocycle_idxs[::-1]  # Flip direction

    # Always start at a nitrogen
    nitrogen_idx = next(iter(carbon_and_nitrogen_idxs))[1]  # Random nitrogen
    nitrogen_loc = macrocycle_idxs.index(nitrogen_idx)
    macrocycle_idxs = macrocycle_idxs[nitrogen_loc:] + macrocycle_idxs[:nitrogen_loc]

    return macrocycle_idxs


def extract_macrocycle(mol: Chem.Mol) -> Chem.Mol:
    macrocycle_idxs = get_macrocycle_idxs(mol)
    if macrocycle_idxs is None:
        raise ValueError(f"No macrocycle detected in '{Chem.MolToSmiles(Chem.RemoveHs(mol))}'")

    macrocycle_idxs = set(macrocycle_idxs)
    to_remove = sorted(
        (atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in macrocycle_idxs),
        reverse=True,
    )

    rwmol = Chem.RWMol(mol)
    for idx in to_remove:
        rwmol.RemoveAtom(idx)

    new_mol = rwmol.GetMol()
    return new_mol


def combine_mols(mols: List[Chem.Mol]) -> Chem.Mol:
    """Combine multiple molecules with one conformer each into one molecule with multiple
    conformers.

    Args:
        mols: List of molecules.

    Returns:
        Combined molecule.
    """
    new_mol = Chem.Mol(mols[0], quickCopy=True)
    for mol in mols:
        conf = Chem.Conformer(mol.GetConformer())
        new_mol.AddConformer(conf, assignId=True)
    return new_mol


def set_atom_positions(
    mol: Chem.Mol,
    xyzs: Union[np.ndarray, pd.DataFrame, List[np.ndarray], List[pd.DataFrame]],
    atom_idxs: Optional[List[int]] = None,
) -> Chem.Mol:
    """Set atom positions of a molecule.

    Args:
        mol: Molecule.
        xyzs: An array of coordinates; a dataframe with 'x', 'y', and 'z' columns (and optionally an index of atom indices); a list of arrays; or a list of dataframes.
        atom_idxs: Atom indices to set atom positions for. Not required if dataframe(s) contain(s) atom indices.

    Returns:
        A copy of the molecule with one conformer for each set of coordinates.
    """
    # If multiple xyxs are provided, make one conformer for each one
    if isinstance(xyzs, (np.ndarray, pd.DataFrame)):
        xyzs = [xyzs]

    if atom_idxs is None:
        assert all(isinstance(xyz, pd.DataFrame) for xyz in xyzs)

    # The positions that don't get set will be the same as in the first conformer of the given mol
    dummy_conf = mol.GetConformer()
    mol = Chem.Mol(mol, quickCopy=True)  # Don't copy conformers

    for xyz in xyzs:
        if isinstance(xyz, pd.DataFrame):
            atom_idxs = xyz.index.tolist()
            xyz = xyz[["x", "y", "z"]].to_numpy()

        xyz = xyz.reshape(-1, 3)

        # Set only the positions at the provided indices
        conf = Chem.Conformer(dummy_conf)
        for atom_idx, pos in zip(atom_idxs, xyz):
            conf.SetAtomPosition(atom_idx, [float(p) for p in pos])

        mol.AddConformer(conf, assignId=True)

    return mol


def dfs(
    root_atom_idx: int,
    mol: Chem.Mol,
    max_depth: int = float("inf"),
    blocked_idxs: Optional[List[int]] = None,
    include_hydrogens: bool = True,
) -> List[int]:
    """Traverse molecular graph with depth-first search from given root atom index.

    Args:
        root_atom_idx: Root atom index.
        mol: Molecule.
        max_depth: Only traverse to this maximum depth.
        blocked_idxs: Don't traverse across these indices. Defaults to None.
        include_hydrogens: Include hydrogen atom indices in returned list.

    Returns:
        List of traversed atom indices in DFS order.
    """
    root_atom = mol.GetAtomWithIdx(root_atom_idx)
    if blocked_idxs is not None:
        blocked_idxs = set(blocked_idxs)
    return _dfs(
        root_atom,
        max_depth=max_depth,
        blocked_idxs=blocked_idxs,
        include_hydrogens=include_hydrogens,
    )


def _dfs(
    atom: Chem.Atom,  # Start from atom so we don't have to get it from index each time
    depth: int = 0,
    max_depth: int = float("inf"),
    blocked_idxs: Optional[Set[int]] = None,
    include_hydrogens: bool = True,
    visited: Optional[Set[int]] = None,
    traversal: Optional[List[int]] = None,
) -> List[int]:
    if visited is None:
        visited = set()
    if traversal is None:
        traversal = []

    if include_hydrogens or atom.GetAtomicNum() != 1:
        atom_idx = atom.GetIdx()
        visited.add(atom_idx)
        traversal.append(atom_idx)

    if depth < max_depth:
        for atom_nei in atom.GetNeighbors():
            atom_nei_idx = atom_nei.GetIdx()
            if atom_nei_idx not in visited:
                if blocked_idxs is None or atom_nei_idx not in blocked_idxs:
                    _dfs(
                        atom_nei,
                        depth=depth + 1,
                        max_depth=max_depth,
                        blocked_idxs=blocked_idxs,
                        include_hydrogens=include_hydrogens,
                        visited=visited,
                        traversal=traversal,
                    )

    return traversal
