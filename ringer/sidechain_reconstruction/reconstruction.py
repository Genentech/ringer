from typing import Dict, Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D

from ringer.utils import chem, peptides, utils

from .transforms import NeRF, TetraPlacer

BACKBONE_BRANCHES = {
    "[C;R:2][CX3;R:1]([N;R:2])=[O:0]": (1.45, np.pi / 2),
    "[CX3;R:2][N;R:1]([C;R:2])-[C;H3:0]": (1.23, np.pi / 2),
}

DEFAULT_OFFSETS = (1, 2, 3)
RINGER_OFFSETS = (0, 1, 1)
NUM_INTERNALS = 3
NUM_SIDECHAIN_INTERNALS = 4


def reindex_internals(array, offset_differences, length: Optional[int] = None):
    """Reindex the array based on the differences in offsets."""
    if length is None:
        length = array.size(1)
    new_index = build_reindex(offset_differences, length)
    return array[:, new_index, torch.arange(array.size(-1))[None, :]]


def build_reindex(offsets, length):
    reindex = (torch.arange(length).reshape(-1, 1) + np.array(offsets)) % length
    return reindex


def convert_offsets(source_offset, target_offset):
    """Convert the difference in offsets to determine how to convert."""
    return tuple(source_offset[i] - target_offset[i] for i in range(NUM_INTERNALS))


def set_rdkit_geometries(
    mol: Chem.Mol, positions: torch.Tensor, add_hydrogens: bool = True, copy: bool = True
) -> Chem.Mol:
    """Set the geometries for the conformers.

    Assumes positions is K x N x 3, where K is the number of conformers, N is the number of atoms,
    and 3 are the cartesian xyz coordinates.
    """
    if copy:
        mol = Chem.Mol(mol, quickCopy=True)  # create an empty mol with no coordinates

    # Get the number of conformers based on the number of positions
    n_conformers = positions.size(0)
    n_atoms = mol.GetNumAtoms()

    # Add an empty conformer with empty positions for n_atoms
    conf = Chem.Conformer(n_atoms)
    mol.AddConformer(conf, assignId=True)

    # just copy over the conformers
    for i in range(n_conformers - 1):  # -1 since it already has one conf
        mol.AddConformer(mol.GetConformer(0), assignId=True)

    for conf_id, xyzs in enumerate(positions):
        conf = mol.GetConformer(conf_id)
        for atom_id in range(n_atoms):
            x, y, z = positions[conf_id][atom_id]
            conf.SetAtomPosition(int(atom_id), Point3D(float(x), float(y), float(z)))
    if add_hydrogens:
        mol = Chem.AddHs(mol, addCoords=True)  # include coords for 3d confs
    return mol


class Macrocycle:
    """A class for processing macrocycles, including calculating indices and structural features
    relevant to peptide geometry and conformations."""

    def __init__(
        self,
        mol: Chem.Mol,
        config: dict,
        coords: bool = False,
        copy: bool = True,
        verify: bool = True,
    ):
        """Initializes the Macrocycle object with a molecule, configuration settings, and options
        for handling coordinates and copying of the molecule.

        Args:
            mol (Chem.Mol): The RDKit molecule object to be processed.
            config (dict): A dictionary containing configuration data and mappings.
            coords (bool, optional): Determines if coordinates are to be used. Defaults to False.
            copy (bool, optional): Determines if a copy of the molecule is to be made. Defaults to True.
            verify (bool): Whether to verify that all atoms are accounted for.
        """
        # Copy the molecule, including the coordinates if there are any, but quickCopy
        # if you don't want the coordinates.
        if copy:
            quick_copy = not coords
            mol = Chem.Mol(mol, quickCopy=quick_copy)

        self.mol = mol
        self.num_atoms = mol.GetNumAtoms()
        self.config = config
        self.amino_acid_df = config["amino_acid_df"]

        if coords:
            assert (
                mol.GetNumConformers() > 0
            ), "Mol must have at least one conformer for coords=True"
            self.positions = np.array([c.GetPositions() for c in mol.GetConformers()])
        else:
            self.position = None

        # Load these in as the lookups based on the sequence
        self.aa_rigid_map_dict = config[
            "aa_rigid_map_dict"
        ]  # dictionary with quadruples for torsions
        self.branched_map_dict = config["branched_map_dict"]

        # Compute the macrocycle index, and also the offset index needed to use NeRF for reconstruction
        self.macrocycle_index = chem.get_macrocycle_idxs(self.mol)
        self.macrocycle_index_offset = self.convert_macrocycle_idxs(
            self.macrocycle_index, offset=3
        )
        self.macrocycle_length = len(self.macrocycle_index)
        self.residue_length = self.macrocycle_length // 3

        # Calculate the indexes
        self.residue_dict = peptides.get_residues(self.mol)
        self.sidechain_tuple_dict = peptides.get_side_chain_torsion_idxs(mol)
        self.rigid_sidechain_dict = self.get_implicit_tuples(
            self.mol, self.residue_dict, self.aa_rigid_map_dict
        )  # also based on get_internal_tuples

        self.branched_residue_dict = self.get_implicit_tuples(
            self.mol, self.residue_dict, self.branched_map_dict
        )  # also based on get_internal_tuples

        # These are pretty straightforward
        (
            self.branched_backbone_index,
            self.branched_backbone_dist_angles,
        ) = self.get_branched_backbones(self.mol)

        self.residue_dict_triples = self.create_residue_dict(
            self.sidechain_tuple_dict, self.residue_dict
        )

        # All the flattened tuples in the right order
        self.macrocycle_tuples = np.array(
            list(utils.get_overlapping_sublists(self.macrocycle_index_offset, 4, wrap=True))
        )
        self.sidechain_tuples = np.array(self.parse_side_chains(self.sidechain_tuple_dict))
        self.rigid_sidechain_tuples = np.array(
            self.get_rigid_sidechain_tuples(self.rigid_sidechain_dict)
        )  # we need this to match the rigid_internals_method
        self.branched_sidechain_tuples = np.array(
            self.get_branched_sidechain_tuples(self.branched_residue_dict)
        )

        # Need these for TetraPlacer
        if self.branched_sidechain_tuples.shape[0] > 0:
            self.stacked_tetrad_tuples = np.vstack(
                [self.branched_sidechain_tuples, self.branched_backbone_index]
            )
        else:
            self.stacked_tetrad_tuples = np.array(self.branched_backbone_index)

        # if there are are no coordinates, then initialize these as None
        if not coords:
            self.aa_internals_dict = config["aa_internals_dict"]
            self.branched_distance_angle_dict = config["branched_distance_angle_dict"]

            # sidechain and macrocycle ring distances
            self.aa_sidechain_dist_dict = config[
                "aa_sidechain_dist_dict"
            ]  # dictionary for lookup of distances
            self.macrocycle_ring_distance_dict = config[
                "macrocycle_ring_distances"
            ]  # lookup for macrocycle distances
            # all the tetrads
            self.branched_sidechain_dist_angles = self.get_branched_sidechain_dist_angles(
                self.residue_dict
            )

            # Need these for NeRF
            self.stacked_tuples = np.vstack(
                [
                    arr
                    for arr in [
                        self.macrocycle_tuples,
                        self.sidechain_tuples,
                        self.rigid_sidechain_tuples,
                    ]
                    if len(arr) > 0
                ]
            )

            self.macrocycle_distances = self.macrocycle_ring_distance_dict[self.residue_length]
            self.sidechain_distances = self.get_sidechain_distances(self.residue_dict_triples)
            self.rigid_sidechain_internals = self.get_rigid_internals(
                self.residue_dict
            )  # this is based on the residue_dict

            self.stacked_tetrad_dist_angles = np.vstack(
                [
                    arr
                    for arr in [
                        self.branched_sidechain_dist_angles,
                        np.array(self.branched_backbone_dist_angles).squeeze(),
                    ]
                    if len(arr) > 0
                ]
            )
        else:
            self.aa_internals_dict = None
            self.branched_distance_angle_dict = None

        if verify:
            self.verify_tuples()

    def verify_tuples(
        self,
    ):
        """Verify that all heavy atom placements are accounted for and present either in the linear
        reconstruction or tetrad reconstruction index."""
        atom_index = set(np.arange(self.mol.GetNumHeavyAtoms()))
        linear = list(self.stacked_tuples[:, -1])
        tetrad = list(self.stacked_tetrad_tuples[:, -1])
        target_atom_ids = set(linear + tetrad)
        diff = atom_index.difference(target_atom_ids)
        assert (
            target_atom_ids == atom_index
        ), f"atom_ids: {diff} not accounted for in the tuple index"

    def create_residue_dict(self, sidechain_tuple_dict: dict, residue_dict: dict) -> dict:
        """Creates a dictionary based on sidechain atom id corresponding to the amino acid
        identity, the mapped atoms, and the relevant sidechain tuples."""
        res = {}
        for sidechain_atom_id, tups in sidechain_tuple_dict.items():
            for frozen_set, letter in residue_dict.items():
                if sidechain_atom_id in frozen_set:
                    res[sidechain_atom_id] = (letter, frozen_set, tups)
        return res

    def parse_side_chains(self, side_chain_dict: dict) -> list:
        side_chain_tuples = []
        for atom_idx, side_chain_idx in side_chain_dict.items():
            side_chain_quads = list(utils.get_overlapping_sublists(side_chain_idx, 4, wrap=False))
            side_chain_tuples.extend(side_chain_quads)
        return side_chain_tuples

    def convert_macrocycle_idxs(self, macrocycle_idxs, offset: int = 3):
        macrocycle_quads = list(macrocycle_idxs).copy()
        for i in range(offset):  # NeRF creates an offset of three
            macrocycle_quads.insert(0, macrocycle_quads.pop())
        return macrocycle_quads

    def get_rigid_sidechain_tuples(self, implicit_side_chain_dict: dict):
        """Process the side chain tuples."""
        implicit_side_chains = []
        for k, v in implicit_side_chain_dict.items():
            implicit_side_chains.extend(v)
        return implicit_side_chains

    def get_branched_sidechain_tuples(self, implicit_side_chain_dict: dict):
        """Process the side chain tuples."""
        implicit_side_chains = []
        for k, v in implicit_side_chain_dict.items():
            implicit_side_chains.extend(v)
        return implicit_side_chains

    def get_bond_ids(self, mol: Chem.Mol, atom_indices: list) -> list:
        """Get the ids for bonds between a set of atoms."""
        bond_ids = []
        for i in atom_indices:
            for b in mol.GetAtoms()[i].GetBonds():
                if b.GetBeginAtomIdx() in atom_indices and b.GetEndAtomIdx() in atom_indices:
                    bond_ids.append(b.GetIdx())
        return list(set(bond_ids))

    def get_rigid_internals(self, residue_dict):
        results = []
        for fs, one_letter_code in residue_dict.items():
            if self.aa_internals_dict.get(one_letter_code) is not None:
                results.extend(self.aa_internals_dict[one_letter_code])
        return np.array(results)

    def get_branched_backbones(self, mol):
        """Get branched carbonyls and N-methyls."""
        indexes = []
        distance_angles = []
        for smarts, distance_angle in BACKBONE_BRANCHES.items():
            matches = self.get_matched_indexes(smarts, mol)
            distance_angle_repeat = len(matches) * [distance_angle]
            indexes.extend(matches)
            if distance_angle_repeat:
                distance_angles.extend(distance_angle_repeat)
        return indexes, distance_angles

    def get_matched_indexes(self, smarts_pattern: str, mol: Chem.Mol):
        match_mol = Chem.MolFromSmarts(smarts_pattern)
        matches = mol.GetSubstructMatches(match_mol)

        return matches

    def get_sidechain_distances(self, residue_dict_triples):
        """Get sidechain distances."""
        results = []
        for sidechain_id, (one_letter_code, fs, tuples) in residue_dict_triples.items():
            results.extend(self.aa_sidechain_dist_dict[one_letter_code])
        return np.array(results)

    def get_branched_sidechain_dist_angles(self, residue_dict):
        results = []
        for fs, one_letter_code in residue_dict.items():
            if self.branched_distance_angle_dict.get(one_letter_code) is not None:
                results.extend(self.branched_distance_angle_dict[one_letter_code])
        return np.array(results)

    def get_implicit_tuples(self, mol, peptide_assignment, map_dict: Dict):
        """Get implicit tuples for a mol and its peptide assignment."""
        amino_acid_tuples = {}
        for i, (frozen_set, one_letter_code) in enumerate(peptide_assignment.items()):
            residue_atom_ids = list(frozen_set)
            bond_ids = list(set(self.get_bond_ids(mol, residue_atom_ids)))

            # This gets the mol to submol map
            mol_to_submol_map = {}
            submol = Chem.PathToSubmol(mol, bond_ids, atomMap=mol_to_submol_map)
            submol_to_mol_map = {v: k for k, v in mol_to_submol_map.items()}

            # This gets the amino acid to submol map
            amino_acid = self.amino_acid_df.loc[one_letter_code].residue_mol
            match = submol.GetSubstructMatch(amino_acid)
            aa_to_submol_mapping = dict(enumerate(match))

            if map_dict.get(one_letter_code) is not None:
                md = map_dict[one_letter_code]
                # print(one_letter_code, md)
                remapped = []
                for tup in md:
                    remapped.append([submol_to_mol_map[aa_to_submol_mapping[i]] for i in tup])

                amino_acid_tuples[f"{one_letter_code}_{i}"] = remapped
        return amino_acid_tuples


class Reconstructor:
    """Reconstruct a macrocycle and its generated conformers.

    Loads in a Macrocycle object that contains the required quadruple indices needed for NeRF,
    along with a set of distances for rotatable side chains, and fixed internals for nonchanging
    rigid sidechains (e.g. phenyl of phenylalanine).
    """

    def __init__(self, macrocycle: Macrocycle, data_type=torch.double):
        self.macrocycle = macrocycle
        self.data_type = data_type
        self.nerf = NeRF(float_type=data_type)
        self.branched_placer = TetraPlacer()

        self.macrocycle_index = macrocycle.macrocycle_index
        self.macrocycle_length = macrocycle.macrocycle_length

        self.macrocycle_tuples = macrocycle.macrocycle_tuples
        self.sidechain_tuples = macrocycle.sidechain_tuples
        self.rigid_sidechain_tuples = macrocycle.rigid_sidechain_tuples

        self.sidechain_atom_ids = list(macrocycle.sidechain_tuple_dict.keys())

        # All the stacked indices
        self.stacked_tuples = torch.LongTensor(macrocycle.stacked_tuples)

        self.macrocycle_distances = macrocycle.macrocycle_distances
        self.sidechain_distances = macrocycle.sidechain_distances
        self.rigid_sidechain_internals = macrocycle.rigid_sidechain_internals

        self.tetrad_indices = torch.LongTensor(macrocycle.stacked_tetrad_tuples)
        self.tetrad_dist_angles = torch.tensor(
            macrocycle.stacked_tetrad_dist_angles, dtype=self.data_type
        )

    def roll_macrocycle(self, internals, quadruples, macrocycle_length):
        """Roll the macrocycle coordinates, both the internals and the indices."""
        cycle = internals[:, :macrocycle_length, :]
        appendages = internals[:, macrocycle_length:, :]

        cycle_quads = quadruples[:macrocycle_length, :]
        appendages_quads = quadruples[macrocycle_length:, :]

        new_cycles = []
        new_quads = []
        for i in range(macrocycle_length):
            rolled_cycles = torch.roll(cycle, shifts=i, dims=1)
            rolled_quads = torch.roll(cycle_quads, shifts=i, dims=0)

            new_cycles.append(torch.cat([rolled_cycles, appendages], dim=1))
            new_quads.append(torch.cat([rolled_quads, appendages_quads], dim=0))

        return torch.stack(new_cycles), torch.stack(new_quads)

    def calculate_min_errors(self, positions: torch.Tensor, macrocycle_length: int):
        """Calculate the deviation from ideal distances generated by sequential reconstruction."""
        norms = []
        for i in range(macrocycle_length):
            norm = torch.norm(
                positions[i][:, 0, :] - positions[i][:, macrocycle_length - 1, :], dim=-1
            )
            norms.append(norm)

        # Get the distance gapped by computing
        # These are the idealized distances based on the bond lengths:
        distance_gaps = np.roll(self.macrocycle_distances, shift=-1)[
            ::-1
        ]  # we are shifting things forward so we need to walk these backwards
        distance_gaps = torch.Tensor(distance_gaps.copy())

        # get the minimum values across all the possible gaps
        # we reshape the distance gaps appropriately
        # mins = (torch.stack(norms) - distance_gaps.reshape(-1,1)).abs().min(dim=0).values
        argmins = (torch.stack(norms) - distance_gaps.reshape(-1, 1)).abs().argmin(dim=0)

        return argmins

    def parse_sidechains(self, sample: Dict, num_conformers: int):
        """Parse side chain information."""
        all_angles = []
        all_torsions = []
        for i, sidechain_atom_id in enumerate(self.sidechain_atom_ids):
            angles = []
            torsions = []
            for j in range(NUM_SIDECHAIN_INTERNALS + 1):  # span sc_0 to sc_4
                if not np.isnan(sample[f"sc_a{j}"][sidechain_atom_id]).all():
                    angles.append(sample[f"sc_a{j}"][sidechain_atom_id])
                    torsions.append(sample[f"sc_chi{j}"][sidechain_atom_id])
            all_angles.append(angles)
            all_torsions.append(torsions)

        all_angles = np.concatenate(all_angles)
        all_torsions = np.concatenate(all_torsions)

        all_distances = np.repeat(self.sidechain_distances[:, np.newaxis], num_conformers, axis=1)

        all_internals = np.dstack([all_distances, all_angles, all_torsions]).transpose(1, 0, 2)

        return all_internals

    def parse_backbone(self, sample: Dict, num_conformers: int):
        """Parse the backbone."""
        # macrocycle_idx = sample['angle'].columns.to_numpy()

        # we can call the standard ring sizes to get these
        dists = np.vstack(num_conformers * [self.macrocycle_distances])
        angles = sample["angle"].to_numpy()
        torsions = sample["dihedral"].to_numpy()

        # these just stack up "RINGER-style" which corresponds to each atom, which we will need
        # to modify to be NeRF style that allows sequential cartesian setting.
        ring_internals = np.stack([dists, angles, torsions]).transpose(1, 2, 0)

        # cast to Tensor
        ring_internals = torch.tensor(ring_internals, dtype=self.data_type)
        # reset these to NeRF style sequentially
        off_diff = convert_offsets(RINGER_OFFSETS, DEFAULT_OFFSETS)
        res = reindex_internals(ring_internals, off_diff)
        return res

    def parse_internals(self, sample: Dict, to_tensor: bool = True):
        """Parse the internal coordinates."""
        num_conformers = len(sample["angle"])

        backbone_internals = self.parse_backbone(sample, num_conformers)
        sidechain_internals = self.parse_sidechains(sample, num_conformers)

        # replicate this a number of times
        if len(self.rigid_sidechain_internals) > 0:
            rigid_sidechain_internals = np.repeat(
                self.rigid_sidechain_internals[:, np.newaxis], num_conformers, axis=1
            ).transpose(1, 0, 2)
            final_internals = np.concatenate(
                [backbone_internals, sidechain_internals, rigid_sidechain_internals], axis=1
            )  # K x N x 3
        else:
            final_internals = np.concatenate(
                [backbone_internals, sidechain_internals], axis=1
            )  # K x N x 3
        if to_tensor:
            final_internals = torch.tensor(final_internals, dtype=self.data_type)
        return final_internals

    def reconstruct(self, internals: torch.tensor, index: torch.tensor) -> torch.Tensor:
        """The main reconstruction function that puts it all together."""
        # Roll the macrocycles to get all possible start positions and go with the lowest error.
        cycles, quads = self.roll_macrocycle(internals, index, self.macrocycle_length)

        # reconstruct cartesians using NeRF
        xyzs = []
        for c, q in zip(cycles, quads):
            xyz = self.nerf.nerf(c, q)
            xyzs.append(xyz)

        # calculate minimal ring error and grab the best indices
        argmins = self.calculate_min_errors(xyzs, self.macrocycle_length)
        stacked_xyzs = torch.stack(xyzs)
        stacked_indices = quads[:, :, -1]
        best_xyzs = stacked_xyzs[argmins, torch.arange(stacked_xyzs.size(1))]
        best_indices = stacked_indices[argmins]

        num_conformers = best_xyzs.shape[0]
        positions = torch.zeros(
            num_conformers, self.macrocycle.num_atoms, 3, dtype=best_xyzs.dtype
        )
        i_indices = torch.arange(positions.size(0)).unsqueeze(1).expand_as(best_indices)
        positions[i_indices, best_indices] = best_xyzs

        # add the tetrads
        positions = self.branched_placer.add_branched_points(
            positions,
            self.tetrad_indices,
            self.tetrad_dist_angles[:, 0],
            self.tetrad_dist_angles[:, 1],
        )

        return positions
