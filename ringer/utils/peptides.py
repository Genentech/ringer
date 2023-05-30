from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import pandas as pd
from rdkit import Chem

from . import chem

AMINO_ACID_DATA_PATH = Path(__file__).resolve().parent / "data/amino_acids.csv"
AMINO_ACID_DATA = pd.read_csv(AMINO_ACID_DATA_PATH, index_col="aa")
AMINO_ACID_DATA["residue_mol"] = AMINO_ACID_DATA["residue_smiles"].map(Chem.MolFromSmiles)

RING_PEPTIDE_BOND_PATTERN = Chem.MolFromSmarts("[C;R:0](=[OX1:1])[C;R:2][N;R:3]")


def get_amino_acid_stereo(symbol: str) -> Optional[str]:
    # None for glycine
    stereo = AMINO_ACID_DATA.loc[symbol]["alpha_carbon_stereo"]
    return stereo if isinstance(stereo, str) else None


def get_residues(
    mol: Chem.Mol,
    residues_in_mol: Optional[List[str]] = None,
    macrocycle_idxs: Optional[List[int]] = None,
) -> Dict[FrozenSet[int], str]:
    """
    Find the residues in a molecule by matching to a known dataset of amino acids.
    Note: This function isn't inherently restricted to macrocycles and would only require little
    tweaking to work for general peptides.

    Args:
        mol: Macrocycle molecule.
        residues_in_mol: If known, this list of residues speeds up the matching process.
        macrocycle_idxs: Atom indices for atoms in the macrocycle backbone.

    Returns:
        Mapping from atom indices in a residue to its residue label.
    """
    if macrocycle_idxs is None:
        macrocycle_idxs = chem.get_macrocycle_idxs(mol)
        if macrocycle_idxs is None:
            raise ValueError(
                f"Couldn't get macrocycle indices for '{Chem.MolToSmiles(Chem.RemoveHs(mol))}'"
            )

    # Note: An alternative to the below algorithm would be to first find all the atom indices in
    # each residue by running DFS from each backbone atom in the residue, extract the residues as
    # new mols, and then match them to the known residues (e.g., by SMILES).
    # If the residues in the molecule are not known a priori, such an approach would be much
    # faster, but because we generally assume that the residue information is available, we use the
    # algorithm below because extracting the residue as a new mol is a little more tedious.

    backbone_idxs = mol.GetSubstructMatches(RING_PEPTIDE_BOND_PATTERN)
    if residues_in_mol is None:
        potential_residues = AMINO_ACID_DATA.index
    else:
        potential_residues = residues_in_mol

    potential_residue_idxs = {}
    for residue in set(potential_residues):
        residue_data = AMINO_ACID_DATA.loc[residue]
        # This might match a partial side chain, e.g., glycine will match all side chains
        # Will match charged and uncharged side chains if the residue SMILES does not have charges
        # Using chirality for this match is very important to distinguish L- and D-amino acids
        residue_matches = mol.GetSubstructMatches(residue_data["residue_mol"], useChirality=True)
        potential_residue_idxs.update({frozenset(match): residue for match in residue_matches})

    # Because we might have partial matches, we need to find all atom indices in each residue in
    # order to compare to the matched residues and get the residue label
    residue_idxs = [
        frozenset(
            side_chain_idx
            for atom_idx in atom_idxs
            for side_chain_idx in chem.dfs(
                atom_idx, mol, blocked_idxs=macrocycle_idxs, include_hydrogens=False
            )
        )
        for atom_idxs in backbone_idxs
    ]

    residue_dict = {}
    for atom_idxs in residue_idxs:
        try:
            residue = potential_residue_idxs[atom_idxs]
        except KeyError:
            raise Exception(
                f"Cannot determine residue for backbone indices '{list(atom_idxs)}' of '{Chem.MolToSmiles(Chem.RemoveHs(mol))}'"
            )
        else:
            residue_dict[atom_idxs] = residue

    return residue_dict
