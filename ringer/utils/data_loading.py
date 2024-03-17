import logging
import multiprocessing
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np

import ringer

from .. import data
from ..data import noised
from . import variance_schedules

INTERNAL_COORDINATES_DEFINITIONS = Literal[
    "distances-angles", "angles", "dihedrals", "angles-sidechains"
]


def get_datasets(
    data_dir: Union[str, Path],
    internal_coordinates_definitions: INTERNAL_COORDINATES_DEFINITIONS = "angles",
    splits: Sequence[str] = ("train", "validation", "test"),
    split_sizes: Sequence[float] = (0.8, 0.1, 0.1),
    use_atom_features: bool = True,
    atom_feature_fingerprint_radius: int = 3,
    atom_feature_fingerprint_size: int = 32,
    max_conf: Union[int, str] = 30,
    timesteps: int = 50,
    weights: Optional[Dict[str, float]] = None,
    variance_schedule: variance_schedules.SCHEDULES = "cosine",
    variance_scale: float = np.pi,
    mask_noise: bool = False,
    mask_noise_for_features: Optional[List[str]] = None,
    exhaustive_t: bool = False,
    use_cache: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    unsafe_cache: bool = False,
    num_proc: int = multiprocessing.cpu_count(),
    sample_seed: int = 42,
) -> Dict[str, noised.NoisedDataset]:
    """Get the dataset objects to use for train/valid/test.

    Note, these need to be wrapped in data loaders later
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        # Assume it's a path relative to the data folder in the top-level directory
        data_dir = ringer.DATA_DIR / data_dir
        if not data_dir.exists():
            raise ValueError(f"Data directory '{data_dir}' doesn't exist")

    clean_dset_class = data.DATASET_CLASSES[internal_coordinates_definitions]
    logging.info(f"Clean dataset class: {clean_dset_class}")

    logging.info(f"Creating data splits: {splits}")
    clean_dset_kwargs = dict(
        data_dir=data_dir,
        use_atom_features=use_atom_features,
        fingerprint_radius=atom_feature_fingerprint_radius,
        fingerprint_size=atom_feature_fingerprint_size,
        split_sizes=split_sizes,
        num_conf=max_conf,
        all_confs_in_test=True,
        weights=weights,
        zero_center=True,
        use_cache=use_cache,
        unsafe_cache=unsafe_cache,
        num_proc=num_proc,
        sample_seed=sample_seed,
    )
    if cache_dir is not None:
        clean_dset_kwargs["cache_dir"] = cache_dir
    clean_dsets = {split: clean_dset_class(split=split, **clean_dset_kwargs) for split in splits}

    # Set the validation set means to the training set means
    if len(clean_dsets) > 1 and clean_dsets["train"].means_dict is not None:
        logging.info(f"Updating validation/test means to {clean_dsets['train'].means}")
        for split, dset in clean_dsets.items():
            if split != "train":
                dset.means = clean_dsets["train"].means_dict

    logging.info(f"Using {noised.NoisedDataset} for noise")
    noised_dsets = {
        split: noised.NoisedDataset(
            dset=dset,
            dset_key="angles",
            timesteps=timesteps,
            exhaustive_t=(split != "train") and exhaustive_t,
            beta_schedule=variance_schedule,
            nonangular_variance=1.0,
            angular_variance=variance_scale,
            mask_noise=mask_noise,
            mask_noise_for_features=mask_noise_for_features,
        )
        for split, dset in clean_dsets.items()
    }
    for split, dset in noised_dsets.items():
        logging.info(f"{split}: {dset}")

    return noised_dsets
