import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from ..utils import utils, variance_schedules
from . import macrocycle


class NoisedDataset(Dataset):
    """Class that produces noised outputs given a wrapped dataset. Wrapped dset should return a
    tensor from __getitem__ if dset_key is not specified; otherwise, returns a dictionary where the
    item to noise is under dset_key.

    modulo can be given as either a float or a list of floats
    """

    def __init__(
        self,
        dset: macrocycle.MacrocycleInternalCoordinateDataset,
        dset_key: str = "angles",
        timesteps: int = 50,
        exhaustive_t: bool = False,
        beta_schedule: variance_schedules.SCHEDULES = "cosine",
        nonangular_variance: float = 1.0,
        angular_variance: float = 1.0,
    ) -> None:
        super().__init__()

        self.dset = dset
        assert hasattr(dset, "feature_names")
        assert hasattr(dset, "feature_is_angular")
        self.dset_key = dset_key
        self.n_features = len(dset.feature_is_angular)

        self.nonangular_var_scale = nonangular_variance
        self.angular_var_scale = angular_variance

        self.timesteps = timesteps
        self.schedule = beta_schedule
        self.exhaustive_timesteps = exhaustive_t
        if self.exhaustive_timesteps:
            logging.info(f"Exhuastive timesteps for {dset}")

        betas = variance_schedules.get_variance_schedule(beta_schedule, timesteps)
        self.alpha_beta_terms = variance_schedules.compute_alphas(betas)

    @property
    def structures(self) -> Optional[Dict[str, Dict[str, pd.DataFrame]]]:
        return self.dset.structures

    @property
    def atom_features(self) -> Optional[Dict[str, Dict[str, Union[Chem.Mol, pd.DataFrame]]]]:
        return self.dset.atom_features

    @property
    def feature_names(self) -> Tuple[str, ...]:
        """Pass through feature names property of wrapped dset."""
        return self.dset.feature_names

    @property
    def feature_is_angular(self) -> Tuple[bool, ...]:
        """Pass through feature is angular property of wrapped dset."""
        return self.dset.feature_is_angular

    @property
    def pad(self) -> int:
        """Pass through the pad property of wrapped dset."""
        return self.dset.pad

    @property
    def means(self) -> Optional[np.ndarray]:
        return self.dset.means

    @property
    def means_dict(self) -> Optional[Dict[str, float]]:
        return self.dset.means_dict

    @means.setter
    def means(self, means: Dict[str, float]) -> None:
        self.dset.means = means

    @property
    def all_lengths(self) -> List[int]:
        return self.dset.all_lengths

    def sample_length(self, *args, **kwargs) -> Union[int, List[int]]:
        return self.dset.sample_length(*args, **kwargs)

    def get_atom_features(
        self, *args, **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[int]]]:
        return self.dset.get_atom_features(*args, **kwargs)

    def get_structure_as_dataframe(self, index: int) -> pd.DataFrame:
        return self.dset.get_structure_as_dataframe(index)

    def __str__(self) -> str:
        return f"NoisedAnglesDataset wrapping {self.dset} with {len(self)} examples with {self.schedule}-{self.timesteps} with variance scales {self.nonangular_var_scale} and {self.angular_var_scale}"

    def __len__(self) -> int:
        if not self.exhaustive_timesteps:
            return len(self.dset)
        else:
            return int(len(self.dset) * self.timesteps)

    def sample_noise(self, vals: torch.Tensor) -> torch.Tensor:
        """Adaptively sample noise based on modulo.

        We scale only the variance because we want the noise to remain zero centered
        """
        # Noise is always 0 centered
        noise = torch.randn_like(vals)

        # Shapes of vals couled be (batch, seq, feat) or (seq, feat)
        # Therefore we need to index into last dimension consistently

        # Scale by provided variance scales based on angular or not
        if self.angular_var_scale != 1.0 or self.nonangular_var_scale != 1.0:
            for j in range(noise.shape[-1]):  # Last dim = feature dim
                s = (
                    self.angular_var_scale
                    if self.feature_is_angular[j]
                    else self.nonangular_var_scale
                )
                noise[..., j] *= s

        # Make sure that the noise doesn't run over the boundaries
        noise[..., self.feature_is_angular] = utils.modulo_with_wrapped_range(
            noise[..., self.feature_is_angular], -np.pi, np.pi
        )

        return noise

    def __getitem__(
        self,
        index: int,
        use_t_val: Optional[int] = None,
        ignore_zero_center: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Gets the i-th item in the dataset and adds noise use_t_val is useful for manually
        querying specific timepoints."""
        assert 0 <= index < len(self), f"Index {index} out of bounds for {len(self)}"
        # Handle cases where we exhaustively loop over t
        if self.exhaustive_timesteps:
            item_index = index // self.timesteps
            assert item_index < len(self)
            time_index = index % self.timesteps
            logging.debug(f"Exhaustive {index} -> item {item_index} at time {time_index}")
            assert (
                item_index * self.timesteps + time_index == index
            ), f"Unexpected indices for {index} -- {item_index} {time_index}"
            item = self.dset.__getitem__(item_index, ignore_zero_center=ignore_zero_center)
        else:
            item = self.dset.__getitem__(index, ignore_zero_center=ignore_zero_center)

        # If wrapped dset returns a dictionary then we extract the item to noise
        if self.dset_key is not None:
            assert isinstance(item, dict)
            vals = item[self.dset_key].clone()
        else:
            vals = item.clone()
        assert isinstance(
            vals, torch.Tensor
        ), f"Using dset_key {self.dset_key} - expected tensor but got {type(vals)}"

        # Sample a random timepoint and add corresponding noise
        if use_t_val is not None:
            assert not self.exhaustive_timesteps, "Cannot use specific t in exhaustive mode"
            t_val = np.clip(np.array([use_t_val]), 0, self.timesteps - 1)
            t = torch.from_numpy(t_val).long()
        elif self.exhaustive_timesteps:
            t = torch.tensor([time_index]).long()  # list to get correct shape
        else:
            t = torch.randint(0, self.timesteps, (1,)).long()

        # Get the values for alpha and beta
        sqrt_alphas_cumprod_t = self.alpha_beta_terms["sqrt_alphas_cumprod"][t.item()]
        sqrt_one_minus_alphas_cumprod_t = self.alpha_beta_terms["sqrt_one_minus_alphas_cumprod"][
            t.item()
        ]
        # Noise is sampled within range of [-pi, pi], and optionally
        # shifted to [0, 2pi] by adding pi
        noise = self.sample_noise(vals)  # Vals passed in only for shape

        # Add noise and ensure noised vals are still in range
        noised_vals = sqrt_alphas_cumprod_t * vals + sqrt_one_minus_alphas_cumprod_t * noise
        assert noised_vals.shape == vals.shape, f"Unexpected shape {noised_vals.shape}"
        # The underlying vals are already shifted, and noise is already shifted
        # All we need to do is ensure we stay on the corresponding manifold
        # Wrap around the correct range
        noised_vals[:, self.feature_is_angular] = utils.modulo_with_wrapped_range(
            noised_vals[:, self.feature_is_angular], -np.pi, np.pi
        )

        retval = {
            "corrupted": noised_vals,
            "t": t,
            "known_noise": noise,
            "sqrt_alphas_cumprod_t": sqrt_alphas_cumprod_t,
            "sqrt_one_minus_alphas_cumprod_t": sqrt_one_minus_alphas_cumprod_t,
        }

        # Update dictionary if wrapped dset returns dicts, else just return
        if isinstance(item, dict):
            assert item.keys().isdisjoint(retval.keys())
            item.update(retval)
            return item
        return retval
