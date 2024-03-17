import functools
import glob
import hashlib
import logging
import multiprocessing
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.nn import functional as F
from torch.utils.data import Dataset

from ..utils import featurization, internal_coords, utils


def _atom_featurization_helper(
    path: Union[str, Path], **kwargs
) -> Union[pd.DataFrame, Tuple[Chem.Mol, pd.DataFrame]]:
    path = Path(path)
    # Assumes path name is sequence of amino acid symbols separated by periods
    residues_in_mol = [aa.strip("[]") for aa in path.stem.replace("Sar", "MeG").split(".")]
    return featurization.featurize_macrocycle_atoms_from_file(
        path, residues_in_mol=residues_in_mol, **kwargs
    )


class MacrocycleInternalCoordinateDataset(Dataset):
    feature_names = ("distance", "angle", "dihedral")
    feature_is_angular = (False, True, True)

    def __init__(
        self,
        data_dir: Union[str, Path],
        use_atom_features: bool = True,
        fingerprint_radius: int = 3,
        fingerprint_size: int = 32,
        split: Optional[Literal["train", "test", "validation"]] = None,
        split_sizes: Sequence[float] = (0.8, 0.1, 0.1),
        num_conf: Union[int, str] = 30,
        all_confs_in_test: bool = True,
        weights: Optional[Dict[str, float]] = None,
        zero_center: bool = True,  # Center the features to have 0 mean
        use_cache: bool = True,  # Use/build cached computations of dihedrals and angles
        cache_dir: Union[str, Path] = Path(os.path.dirname(os.path.abspath(__file__))),
        unsafe_cache: bool = False,  # Don't check codebase hash
        num_proc: int = multiprocessing.cpu_count(),
        seed: int = 6489,
        sample_seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Path to data directory containing one pickle file per mol. Each pickle file should be the sequence of amino acid symbols separated by periods.
            use_atom_features: Needed for training a model conditioned on atom sequence.
            fingerprint_radius: Morgan fingerprint radius used to featurize side chain attached to each atom.
            fingerprint_size: Morgan fingerprint size used to featurize side chain attached to each atom.
            split: Which split to load. The pickle files are shuffled deterministically before splitting and the size of the given split is determined according to split_sizes.
            split_sizes: Fractions of the data to use for train, validation, and test splits.
            num_conf: Maximum number of conformers to load for each molecule (can be "all").
            all_confs_in_test: Override num_conf if split is test to load all conformers for each molecule.
            weights: For features with weights other than 1, specify their weights as a mapping from feature name to their weight.
            seed: Random seed needed to reproduce data splitting so that loading from the same files in the same directory produces identical splits.
            sample_seed: Random seed used for sampling sequence lengths.
        """
        super().__init__()

        # gather files
        self.data_src = str(data_dir)
        fnames = self.__get_fnames(Path(data_dir))
        self.fnames = fnames

        if len(split_sizes) != 3:
            raise ValueError("split_sizes must have a value for train, validation, and test")
        if sum(split_sizes) != 1:
            raise ValueError("Split sizes do not sum to 1")

        self.rng = np.random.default_rng(seed=seed)
        # Shuffle the sequences so contiguous splits acts like random splits
        self.rng.shuffle(self.fnames)
        if split is not None:
            split_idx = int(len(self.fnames) * split_sizes[0])
            if split == "train":
                self.fnames = self.fnames[:split_idx]
                self.split_size = split_sizes[0]
            elif split == "validation":
                self.fnames = self.fnames[
                    split_idx : split_idx + int(len(self.fnames) * split_sizes[1])
                ]
                self.split_size = split_sizes[1]
            elif split == "test":
                self.fnames = self.fnames[split_idx + int(len(self.fnames) * split_sizes[1]) :]
                self.split_size = split_sizes[2]
            else:
                raise ValueError(f"Unknown split: {split}")

            logging.info(f"Split {split} contains {len(self.fnames)} molecules")

            if all_confs_in_test and split == "test":
                self.num_conf = "all"
            else:
                self.num_conf = num_conf
        else:
            self.num_conf = num_conf
            self.split_size = None

        self.split = split

        if weights is not None:
            for name in weights:
                assert name in self.feature_names
        self.weights = weights

        self.use_atom_features = use_atom_features
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_size = fingerprint_size
        self.atom_features: Optional[Dict[str, Dict[str, Union[Chem.Mol, pd.DataFrame]]]] = None
        # TODO: Load mols first, then compute featurization from mol

        # self.structures should be a dict of dicts with keys (distance, angle, dihedral)
        # Define as None by default; allow for easy checking later
        self.structures: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None
        codebase_hash = utils.md5_all_py_files(os.path.dirname(os.path.abspath(__file__)))
        # Default to false; assuming no cache, also doesn't match
        codebase_matches_hash = False
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        if use_cache and os.path.exists(self.cache_fname):
            logging.info(f"Loading cached dataset from {self.cache_fname}")
            loaded_hash, loaded_structures = self.__load_cache()
            codebase_matches_hash = loaded_hash == codebase_hash
            if not unsafe_cache and not codebase_matches_hash:
                logging.warning(
                    "Mismatched hashes between codebase and cached values; updating cached values"
                )
            else:
                self.structures = loaded_structures
                logging.info("Hash matches between codebase and cached values!")
        codebase_matches_hash_features = False
        if self.use_atom_features and use_cache and os.path.exists(self.atom_feature_cache_fname):
            logging.info(f"Loading cached features from {self.atom_feature_cache_fname}")
            loaded_hash, loaded_atom_features = self.__load_atom_feature_cache()
            codebase_matches_hash_features = loaded_hash == codebase_hash
            if not unsafe_cache and not codebase_matches_hash_features:
                logging.warning(
                    "Mismatched hashes between codebase and cached values; updating cached values"
                )
            else:
                self.atom_features = loaded_atom_features
                logging.info("Hash matches between codebase and cached values!")
        # We have not yet populated self.structures
        if self.structures is None:
            self.__clean_mismatched_caches()
            self.structures = self._compute_featurization(self.fnames, num_proc=num_proc)
            if use_cache and not codebase_matches_hash:
                logging.info(f"Saving full dataset to cache at {self.cache_fname}")
                self.__save_cache(codebase_hash)
        if self.use_atom_features and self.atom_features is None:
            self.__clean_mismatched_feature_caches()
            self.atom_features = self.__compute_atom_featurization(self.fnames, num_proc=num_proc)
            if use_cache and not codebase_matches_hash_features:
                logging.info(f"Saving features to cache at {self.atom_feature_cache_fname}")
                self.__save_feature_cache(codebase_hash)

        # Determine padding based on longest sequence in the data. There's a small chance that one
        # of the splits doesn't have an example with the longest sequence length, but generally we
        # should be safe.
        # Also get a flattened list over all conformers so that self.__getitem__ only returns the
        # internal coordinates for a single conformer. For this, we need the mapping from index to
        # (fname, conf_idx).
        self.pad = 0
        self.global_conf_ids = []  # Store list of (fname, conf_idx)
        curr_idx = 0
        for fname, structure in self.structures.items():
            distances = structure["distance"]
            assert len(distances) == len(structure["angle"]) == len(structure["dihedral"])
            self.pad = max(len(distances.columns), self.pad)
            for conf_idx in range(len(distances.index)):
                self.global_conf_ids.append((fname, conf_idx))
                curr_idx += 1

        logging.info(f"Structures contain {len(self.global_conf_ids)} conformers")

        # if given, zero center the features
        self._means: Optional[Dict[str, float]] = None
        if zero_center:
            # Note that there is no padding yet
            self._means = {}
            for feature_name, is_angular in zip(self.feature_names, self.feature_is_angular):
                all_features = np.concatenate(
                    [s[feature_name].to_numpy().ravel() for s in self.structures.values()]
                )
                mean_func = utils.wrapped_mean if is_angular else np.nanmean
                feature_mean = mean_func(all_features)
                self._means[feature_name] = feature_mean

            # Subtract the mean and perform modulo where values are radial (note that we
            # actually defer this operation until self.__getitem__)
            logging.info(f"Offsetting features {self.feature_names} by means {self.means}")

        # Compute the mean internal coordinates by atom (bond) type
        # TODO: Simplify this
        self.atom_type_means = {}
        self.atom_type_stdevs = {}
        for feature_name in MacrocycleInternalCoordinateDataset.feature_names:
            sums = defaultdict(float)
            counts = defaultdict(int)

            for structure in self.structures.values():
                ics = structure[feature_name].copy()
                ics.columns = structure["atom_labels"]

                for atom_label in internal_coords.BACKBONE_ATOM_LABELS:
                    ic_array = ics[atom_label].to_numpy()
                    sums[atom_label] += ic_array.sum()
                    counts[atom_label] += ic_array.size

            self.atom_type_means[feature_name] = {
                label: sums[label] / counts[label]
                for label in internal_coords.BACKBONE_ATOM_LABELS
            }
        for feature_name in MacrocycleInternalCoordinateDataset.feature_names:
            sq_dev_sums = defaultdict(float)
            counts = defaultdict(int)

            for structure in self.structures.values():
                ics = structure[feature_name].copy()
                ics.columns = structure["atom_labels"]

                for atom_label in internal_coords.BACKBONE_ATOM_LABELS:
                    ic_array = ics[atom_label].to_numpy()
                    sq_dev_sums[atom_label] += (
                        (ic_array - self.atom_type_means[feature_name][atom_label]) ** 2
                    ).sum()
                    counts[atom_label] += ic_array.size

            self.atom_type_stdevs[feature_name] = {
                label: np.sqrt(sq_dev_sums[label] / counts[label])
                for label in internal_coords.BACKBONE_ATOM_LABELS
            }

        # Aggregate lengths
        self.all_lengths = [len(s["distance"].columns) for s in self.structures.values()]
        self._length_rng = np.random.default_rng(seed=sample_seed)
        logging.info(
            f"Length of angles: {np.min(self.all_lengths)}-{np.max(self.all_lengths)}, mean {np.mean(self.all_lengths)}"
        )

    def __get_fnames(self, data_dir: Path, ext: str = ".pickle") -> List[str]:
        return [str(path.resolve()) for path in data_dir.glob(f"*{ext}")]

    @property
    def cache_fname(self) -> str:
        """Return the filename for the cache file."""
        if os.path.isdir(self.data_src):
            k = os.path.basename(self.data_src)
        else:
            k = self.data_src

        # Create md5 of all the filenames (NOT their contents)
        hash_md5 = hashlib.md5()
        for fname in self.fnames:
            hash_md5.update(os.path.basename(fname).encode())
        filename_hash = hash_md5.hexdigest()

        cache_fname = os.path.join(
            self.cache_dir,
            f"cache_canonical_macrocycle_structures_{k}_{self.num_conf}conf_{filename_hash}",
        )
        if self.split is not None:
            cache_fname = f"{cache_fname}_{self.split}{self.split_size}.pickle"
        else:
            cache_fname = f"{cache_fname}.pickle"

        return cache_fname

    @property
    def atom_feature_cache_fname(self) -> str:
        """Return the filename for the feature cache file."""
        if os.path.isdir(self.data_src):
            k = os.path.basename(self.data_src)
        else:
            raise NotImplementedError(f"'{self.data_src} must be a valid directory")

        # Create md5 of all the filenames (NOT their contents)
        hash_md5 = hashlib.md5()
        for fname in self.fnames:
            hash_md5.update(os.path.basename(fname).encode())
        filename_hash = hash_md5.hexdigest()

        cache_fname = os.path.join(
            self.cache_dir,
            f"cache_macrocycle_features_{k}_fprad{self.fingerprint_radius}_fpsize{self.fingerprint_size}_{filename_hash}",
        )
        if self.split is not None:
            cache_fname = f"{cache_fname}_{self.split}{self.split_size}.pickle"
        else:
            cache_fname = f"{cache_fname}.pickle"

        return cache_fname

    def __clean_mismatched_caches(self) -> None:
        """Clean out mismatched cache files."""
        if not self.use_cache:
            logging.info("Not using cache -- skipping cache cleaning")
            return

        if os.path.isdir(self.data_src):
            k = os.path.basename(self.data_src)
        else:
            k = self.data_src

        cache_pattern = os.path.join(
            self.cache_dir,
            f"cache_canonical_macrocycle_structures_{k}_{self.num_conf}conf_*",
        )
        if self.split is not None:
            cache_pattern = f"{cache_pattern}_{self.split}{self.split_size}.pickle"
        else:
            cache_pattern = f"{cache_pattern}.pickle"

        matches = glob.glob(cache_pattern)
        if not matches:
            logging.info(f"No cache files found matching {matches}, no cleaning necessary")
        for fname in matches:
            if fname != self.cache_fname:
                logging.info(f"Removing old cache file {fname}")
                os.remove(fname)

    def __clean_mismatched_feature_caches(self) -> None:
        """Clean out mismatched feature cache files."""
        if not self.use_cache:
            logging.info("Not using cache -- skipping cache cleaning")
            return

        if os.path.isdir(self.data_src):
            k = os.path.basename(self.data_src)
        else:
            raise NotImplementedError(f"'{self.data_src} must be a valid directory")

        cache_pattern = os.path.join(
            self.cache_dir,
            f"cache_macrocycle_features_{k}_fprad{self.fingerprint_radius}_fpsize{self.fingerprint_size}_*",
        )
        if self.split is not None:
            cache_pattern = f"{cache_pattern}_{self.split}{self.split_size}.pickle"
        else:
            cache_pattern = f"{cache_pattern}.pickle"

        matches = glob.glob(cache_pattern)
        if not matches:
            logging.info(f"No feature cache files found matching {matches}, no cleaning necessary")
        for fname in matches:
            if fname != self.cache_fname:
                logging.info(f"Removing old feature cache file {fname}")
                os.remove(fname)

    def __save_cache(self, codebase_hash: str) -> None:
        with open(self.cache_fname, "wb") as sink:
            pickle.dump(
                (codebase_hash, self.structures),
                sink,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def __save_feature_cache(self, codebase_hash: str) -> None:
        with open(self.atom_feature_cache_fname, "wb") as sink:
            pickle.dump(
                (codebase_hash, self.atom_features),
                sink,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def __load_cache(self) -> Tuple[str, Dict[str, Dict[str, pd.DataFrame]]]:
        with open(self.cache_fname, "rb") as source:
            loaded_hash, loaded_structures = pickle.load(source)
        return loaded_hash, loaded_structures

    def __load_atom_feature_cache(
        self,
    ) -> Tuple[str, Dict[str, Dict[str, Union[Chem.Mol, pd.DataFrame]]]]:
        with open(self.atom_feature_cache_fname, "rb") as source:
            loaded_hash, loaded_atom_features = pickle.load(source)
        return loaded_hash, loaded_atom_features

    def _compute_featurization(
        self, fnames: Sequence[str], num_proc: int = 1
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        pfunc = internal_coords.get_macrocycle_distances_and_angles_from_file

        logging.info(f"Computing full dataset of {len(fnames)} with {num_proc} processes")
        if num_proc == 1:
            internal_coord_dicts = list(map(pfunc, fnames))
        else:
            with multiprocessing.Pool(processes=num_proc) as pool:
                internal_coord_dicts = list(pool.map(pfunc, fnames))

        structures = {}
        for fname, ic_dict in zip(fnames, internal_coord_dicts):
            if isinstance(self.num_conf, int):
                for k, v in ic_dict.items():
                    # Can't use self.feature_names here
                    if k in MacrocycleInternalCoordinateDataset.feature_names:
                        ic_dict[k] = v.iloc[: self.num_conf]
            structures[fname] = ic_dict

        return structures

    def __compute_atom_featurization(
        self, fnames: Sequence[str], num_proc: int = 1
    ) -> Dict[str, Dict[str, Union[Chem.Mol, pd.DataFrame]]]:
        pfunc = functools.partial(
            _atom_featurization_helper,
            radius=self.fingerprint_radius,
            size=self.fingerprint_size,
            return_mol=self.split != "train",
        )

        logging.info(f"Computing atom features from {len(fnames)} files with {num_proc} processes")
        if num_proc == 1:
            mols_and_features = list(map(pfunc, fnames))
        else:
            with multiprocessing.Pool(processes=num_proc) as pool:
                mols_and_features = list(pool.map(pfunc, fnames))

        if self.split == "train":
            features = {
                fname: {"atom_features": feat_df}
                for fname, feat_df in zip(fnames, mols_and_features)
            }
        else:
            features = {
                fname: {"mol": mol, "atom_features": feat_df}
                for fname, (mol, feat_df) in zip(fnames, mols_and_features)
            }
        return features

    def sample_length(self, n: int = 1) -> Union[int, List[int]]:
        """Sample a observed length of a sequence."""
        assert n > 0
        if n == 1:
            length = self._length_rng.choice(self.all_lengths)
        else:
            length = self._length_rng.choice(self.all_lengths, size=n, replace=True).tolist()
        return length

    def get_atom_features(
        self,
        fname: str,
        pad: bool = True,
        atom_idxs: Optional[List[int]] = None,
        return_idxs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        """Get the atom features for the given key (fname) and optionally pad and/or reorder them.

        Args:
            fname: Key to get features from self.atom_features dictionary.
            pad: Pad features. Defaults to True.
            atom_idxs: Reorder features according to these atom indices. Defaults to None.
            return_idxs: Return atom indices. If atom_idxs is None, return the original ones. Defaults to False.

        Returns:
            Tensor of atom features with shape [seq_len x num_feat] and (optionally) atom indices.
        """
        atom_features = self.atom_features[fname]["atom_features"]

        if atom_idxs is None:
            atom_idxs = atom_features.index.tolist()

        atom_features = atom_features.loc[atom_idxs].to_numpy()
        atom_features = torch.from_numpy(atom_features).float()

        if pad:
            atom_features = F.pad(
                atom_features,
                (0, 0, 0, self.pad - len(atom_features)),
                mode="constant",
                value=0.0,
            )

        if return_idxs:
            return atom_features, atom_idxs

        return atom_features

    def get_structure_as_dataframe(self, index: int) -> pd.DataFrame:
        fname, conf_idx = self.global_conf_ids[index]
        structure = self.structures[fname]

        internal_coords_allconfs = [structure[name] for name in self.feature_names]
        structure_df = pd.DataFrame(
            [df.loc[conf_idx] for df in internal_coords_allconfs], index=self.feature_names
        ).T
        structure_df["atom_label"] = structure["atom_labels"]

        return structure_df

    @property
    def means(self) -> Optional[np.ndarray]:
        # Extracts only the relevant means for child classes that override cls.feature_names
        if self._means is None:
            return None
        return np.array([self._means[name] for name in self.feature_names])

    @property
    def means_dict(self) -> Optional[Dict[str, float]]:
        return self._means

    @means.setter
    def means(self, means: Dict[str, float]) -> None:
        if any(k not in self.feature_names for k in means.keys()):
            raise ValueError(
                f"Expected dictionary of '{self.feature_names}' means, received: '{means}'"
            )
        self._means = means

    def __len__(self) -> int:
        return len(self.global_conf_ids)

    def __getitem__(self, index: int, ignore_zero_center: bool = False) -> Dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        fname, conf_idx = self.global_conf_ids[index]
        structure = self.structures[fname]

        internal_coords_allconfs = [structure[name] for name in self.feature_names]
        macrocycle_idxs = internal_coords_allconfs[0].columns
        assert all((df.columns == macrocycle_idxs).all() for df in internal_coords_allconfs)
        macrocycle_idxs = list(macrocycle_idxs)

        internal_coords = np.vstack(
            [df.loc[conf_idx].to_numpy() for df in internal_coords_allconfs]
        ).T

        means = self.means
        if means is not None and not ignore_zero_center:
            internal_coords = internal_coords - means
            internal_coords[:, self.feature_is_angular] = utils.modulo_with_wrapped_range(
                internal_coords[:, self.feature_is_angular], -np.pi, np.pi
            )

        # Replace nan values with zero
        np.nan_to_num(internal_coords, copy=False, nan=0)

        # Create attention mask. 0 indicates masked
        seq_length = min(self.pad, len(internal_coords))
        attn_mask = torch.zeros(size=(self.pad,))
        attn_mask[:seq_length] = 1.0

        # Perform padding
        if len(internal_coords) < self.pad:
            internal_coords = np.pad(
                internal_coords,
                ((0, self.pad - len(internal_coords)), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        # Create position IDs
        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)

        # Create backbone atom IDs
        atom_ids = torch.zeros(self.pad, dtype=torch.long)
        atom_ids[: len(structure["atom_ids"])] = torch.tensor(
            structure["atom_ids"], dtype=torch.long
        )

        assert utils.tolerant_comparison_check(
            internal_coords[:, self.feature_is_angular], ">=", -np.pi
        ), f"Illegal value: {np.min(internal_coords[:, self.feature_is_angular])}"
        assert utils.tolerant_comparison_check(
            internal_coords[:, self.feature_is_angular], "<=", np.pi
        ), f"Illegal value: {np.max(internal_coords[:, self.feature_is_angular])}"
        angles = torch.from_numpy(internal_coords).float()

        weights = torch.ones(len(self.feature_names)).float()
        if self.weights is not None:
            feat_to_idx = dict(zip(self.feature_names, range(len(self.feature_names))))
            for feat_name, weight in self.weights.items():
                weights[feat_to_idx[feat_name]] = weight

        retval = {
            "angles": angles,
            "attn_mask": attn_mask,
            "position_ids": position_ids,
            "atom_ids": atom_ids,
            "lengths": torch.tensor(seq_length, dtype=torch.int64),
            "weights": weights,
        }

        if self.use_atom_features:
            atom_features = self.get_atom_features(
                fname, pad=True, atom_idxs=macrocycle_idxs, return_idxs=False
            )
            retval["atom_features"] = atom_features

        return retval


class MacrocycleAnglesDataset(MacrocycleInternalCoordinateDataset):
    feature_names = ("angle", "dihedral")
    feature_is_angular = (True, True)


class MacrocycleDihedralsDataset(MacrocycleInternalCoordinateDataset):
    feature_names = ("dihedral",)
    feature_is_angular = (True,)


class MacrocycleAnglesWithSideChainsDataset(MacrocycleInternalCoordinateDataset):
    side_chain_feature_names = {
        "angle": ("sc_a0", "sc_a1", "sc_a2", "sc_a3", "sc_a4"),
        "dihedral": ("sc_chi0", "sc_chi1", "sc_chi2", "sc_chi3", "sc_chi4"),
    }
    feature_names = (
        ("angle", "dihedral")
        + side_chain_feature_names["angle"]
        + side_chain_feature_names["dihedral"]
    )
    feature_is_angular = (True,) * len(feature_names)
    feature_is_sidechain = (False, False) + (True,) * (
        len(side_chain_feature_names["angle"]) + len(side_chain_feature_names["dihedral"])
    )

    @classmethod
    def _flatten_side_chain_features(
        cls, internal_coord_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        # Atom indices and number of confs should be the same for each coordinate
        atom_idxs = list(internal_coord_dict[cls.feature_names[0]].columns)
        total_confs = len(internal_coord_dict[cls.feature_names[0]])

        # Modify in place
        for feature_type, feature_names in cls.side_chain_feature_names.items():
            for position, feature_name in enumerate(feature_names):
                side_chain_data = {}

                for atom_idx in atom_idxs:
                    if atom_idx in internal_coord_dict["side_chains"]:
                        side_chain_ic_df = internal_coord_dict["side_chains"][atom_idx][
                            feature_type
                        ]

                        try:
                            side_chain_ics = side_chain_ic_df.iloc[:, position]
                        except IndexError:
                            side_chain_ics = pd.Series(np.nan, index=range(total_confs))
                    else:
                        side_chain_ics = pd.Series(np.nan, index=range(total_confs))

                    side_chain_data[atom_idx] = side_chain_ics

                side_chain_df = pd.DataFrame(side_chain_data)
                side_chain_df.index.name = "conf_idx"
                internal_coord_dict[feature_name] = side_chain_df

        return internal_coord_dict

    def _compute_featurization(
        self, fnames: Sequence[str], num_proc: int = 1
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        pfunc = functools.partial(
            internal_coords.get_macrocycle_distances_and_angles_from_file, include_side_chains=True
        )

        logging.info(f"Computing full dataset of {len(fnames)} with {num_proc} processes")
        if num_proc == 1:
            internal_coord_dicts = list(map(pfunc, fnames))
        else:
            with multiprocessing.Pool(processes=num_proc) as pool:
                internal_coord_dicts = list(pool.map(pfunc, fnames))

        structures = {}
        for fname, ic_dict in zip(fnames, internal_coord_dicts):
            ic_dict = self._flatten_side_chain_features(ic_dict)

            if isinstance(self.num_conf, int):
                for k, v in ic_dict.items():
                    if (
                        k in MacrocycleInternalCoordinateDataset.feature_names
                        or k in self.feature_names
                    ):
                        ic_dict[k] = v.iloc[: self.num_conf]

            structures[fname] = ic_dict

        return structures

    # TODO: use __getitem__ from parent instead, just copying now for quick prototyping
    def __getitem__(self, index: int, ignore_zero_center: bool = False) -> Dict[str, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")

        fname, conf_idx = self.global_conf_ids[index]
        structure = self.structures[fname]

        internal_coords_allconfs = [structure[name] for name in self.feature_names]
        macrocycle_idxs = internal_coords_allconfs[0].columns
        assert all((df.columns == macrocycle_idxs).all() for df in internal_coords_allconfs)
        macrocycle_idxs = list(macrocycle_idxs)

        internal_coords = np.vstack(
            [df.loc[conf_idx].to_numpy() for df in internal_coords_allconfs]
        ).T

        means = self.means
        if means is not None and not ignore_zero_center:
            internal_coords = internal_coords - means
            internal_coords[:, self.feature_is_angular] = utils.modulo_with_wrapped_range(
                internal_coords[:, self.feature_is_angular], -np.pi, np.pi
            )

        # Create mask to mask out unused side-chain features. 0 indicates masked
        seq_length = min(self.pad, len(internal_coords))
        feat_mask = torch.zeros(size=(self.pad, len(self.feature_names)))
        feat_mask[:seq_length] = ~torch.from_numpy(np.isnan(internal_coords))

        # Replace nan values with zero
        np.nan_to_num(internal_coords, copy=False, nan=0)

        # Create attention mask. 0 indicates masked
        attn_mask = torch.zeros(size=(self.pad,))
        attn_mask[:seq_length] = 1.0

        # Perform padding
        if len(internal_coords) < self.pad:
            internal_coords = np.pad(
                internal_coords,
                ((0, self.pad - len(internal_coords)), (0, 0)),
                mode="constant",
                constant_values=0.0,
            )

        # Create position IDs
        position_ids = torch.arange(start=0, end=self.pad, step=1, dtype=torch.long)

        # Create backbone atom IDs
        atom_ids = torch.zeros(self.pad, dtype=torch.long)
        atom_ids[: len(structure["atom_ids"])] = torch.tensor(
            structure["atom_ids"], dtype=torch.long
        )

        assert utils.tolerant_comparison_check(
            internal_coords[:, self.feature_is_angular], ">=", -np.pi
        ), f"Illegal value: {np.min(internal_coords[:, self.feature_is_angular])}"
        assert utils.tolerant_comparison_check(
            internal_coords[:, self.feature_is_angular], "<=", np.pi
        ), f"Illegal value: {np.max(internal_coords[:, self.feature_is_angular])}"
        angles = torch.from_numpy(internal_coords).float()

        weights = torch.ones(len(self.feature_names)).float()
        if self.weights is not None:
            feat_to_idx = dict(zip(self.feature_names, range(len(self.feature_names))))
            for feat_name, weight in self.weights.items():
                weights[feat_to_idx[feat_name]] = weight

        retval = {
            "angles": angles,
            "attn_mask": attn_mask,
            "feat_mask": feat_mask,
            "position_ids": position_ids,
            "atom_ids": atom_ids,
            "lengths": torch.tensor(seq_length, dtype=torch.int64),
            "weights": weights,
        }

        if self.use_atom_features:
            atom_features = self.get_atom_features(
                fname, pad=True, atom_idxs=macrocycle_idxs, return_idxs=False
            )
            retval["atom_features"] = atom_features

        return retval
