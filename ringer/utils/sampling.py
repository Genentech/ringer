import logging
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from ..data import noised
from . import internal_coords, utils, variance_schedules


@torch.no_grad()
def p_sample(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    seq_lengths: Sequence[int],
    t_index: Union[int, torch.Tensor],
    betas: torch.Tensor,
    atom_ids: Optional[torch.Tensor] = None,
    atom_features: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sample the given timestep.

    Note that this _may_ fall off the manifold if we just feed the output back into itself
    repeatedly, so we need to perform modulo on it (see p_sample_loop)
    """
    # Calculate alphas and betas
    alpha_beta_values = variance_schedules.compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])

    # Select based on time
    t_unique = torch.unique(t)
    assert len(t_unique) == 1, f"Got multiple values for t: {t_unique}"
    t_index = t_unique.item()
    sqrt_recip_alphas_t = sqrt_recip_alphas[t_index]
    betas_t = betas[t_index]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values["sqrt_one_minus_alphas_cumprod"][t_index]

    # Create the attention mask
    attn_mask = torch.zeros(x.shape[:2], device=x.device)
    for i, seq_length in enumerate(seq_lengths):
        attn_mask[i, :seq_length] = 1.0

    # Use model to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x
        - betas_t
        * model(x, t, attention_mask=attn_mask, atom_ids=atom_ids, atom_features=atom_features)
        / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = alpha_beta_values["posterior_variance"][t_index]
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    seq_lengths: Sequence[int],
    noise: torch.Tensor,
    timesteps: int,
    betas: torch.Tensor,
    atom_ids: Optional[torch.Tensor] = None,
    atom_features: Optional[torch.Tensor] = None,
    is_angle: Union[bool, Sequence[bool]] = (False, True, True),
    disable_pbar: bool = False,
) -> torch.Tensor:
    """Returns a tensor of shape [timesteps x batch_size x seq_len x num_feat]"""
    device = next(model.parameters()).device
    b = noise.shape[0]
    x = noise.to(device)
    # Report metrics on starting noise
    # amin and amax support reducing on multiple dimensions
    logging.info(
        f"Starting from noise {noise.shape} with angularity {is_angle} and range {torch.amin(x, dim=(0, 1))} - {torch.amax(x, dim=(0, 1))} using {device}"
    )

    outputs = []
    for i in tqdm(
        reversed(range(0, timesteps)), desc="Time step", total=timesteps, disable=disable_pbar
    ):
        # Shape is (batch, seq_len, num_output)
        x = p_sample(
            model=model,
            x=x,
            t=torch.full((b,), i, device=device, dtype=torch.long),  # Time vector
            seq_lengths=seq_lengths,
            t_index=i,
            betas=betas,
            atom_ids=None if atom_ids is None else atom_ids.to(device),
            atom_features=None if atom_features is None else atom_features.to(device),
        )

        # Wrap if angular
        if isinstance(is_angle, bool):
            if is_angle:
                x = utils.modulo_with_wrapped_range(x, range_min=-torch.pi, range_max=torch.pi)
        else:
            assert len(is_angle) == x.shape[-1]
            x[..., is_angle] = utils.modulo_with_wrapped_range(
                x[..., is_angle], range_min=-torch.pi, range_max=torch.pi
            )

        outputs.append(x.cpu())

    return torch.stack(outputs)


def sample_batch(
    model: nn.Module,
    dset: noised.NoisedDataset,
    seq_lengths: Sequence[int],
    atom_ids: Optional[torch.Tensor] = None,
    atom_features: Optional[torch.Tensor] = None,
    uniform: bool = False,
    final_timepoint_only: bool = True,
    disable_pbar: bool = False,
) -> List[np.ndarray]:
    noise = dset.sample_noise(
        torch.zeros((len(seq_lengths), dset.pad, model.n_inputs), dtype=torch.float32),
        uniform=uniform,
    )

    samples = p_sample_loop(
        model=model,
        seq_lengths=seq_lengths,
        noise=noise,
        timesteps=dset.timesteps,
        betas=dset.alpha_beta_terms["betas"],
        atom_ids=atom_ids,
        atom_features=atom_features,
        is_angle=dset.feature_is_angular,
        disable_pbar=disable_pbar,
    )  # [timesteps x batch_size x seq_len x num_feat]

    if final_timepoint_only:
        samples = samples[-1]

    # Assumes dset.means contains the training data means
    means = dset.means
    if means is not None:
        logging.info(f"Shifting predicted values by original offset: {means}")
        samples += means
        # Wrap because shifting could have gone beyond boundary
        samples[..., dset.feature_is_angular] = utils.modulo_with_wrapped_range(
            samples[..., dset.feature_is_angular], range_min=-torch.pi, range_max=torch.pi
        )

    # Trim each element in the batch to its sequence length
    trimmed_samples = [
        samples[..., i, :seq_len, :].numpy() for i, seq_len in enumerate(seq_lengths)
    ]

    return trimmed_samples


def sample_unconditional_from_lengths(
    model: nn.Module,
    dset: noised.NoisedDataset,
    seq_lengths: Sequence[int],
    uniform: bool = False,
    batch_size: int = 65536,
    final_timepoint_only: bool = True,
    disable_pbar: bool = False,
) -> Union[List[np.ndarray], List[pd.DataFrame]]:
    """Run reverse diffusion for unconditional macrocycle backbone generation.

    Args:
        model: Model.
        dset: Only needed for its means, sample_noise, timesteps, alpha_beta_terms, feature_is_angular, and pad attributes.
        seq_lengths: Generate one sample for each sequence length provided.
        uniform: Sample uniformly instead of from a wrapped normal.
        batch_size: Batch size.
        final_timepoint_only: Only return the sample at the final (non-noisy) timepoint.
        disable_pbar: Don't display a progress bar.
    """
    samples = []
    atom_id_lists = []
    chunks = [(i, i + batch_size) for i in range(0, len(seq_lengths), batch_size)]

    logging.info(f"Sampling {len(seq_lengths)} items in batches of size {batch_size}")
    for idx_start, idx_end in chunks:
        seq_lengths_batch = seq_lengths[idx_start:idx_end]

        # Need backbone atom labels
        # Technically, we don't need to do this separately for each sequence because the attention
        # mask will take care of the extraneous ones
        atom_ids_batch = torch.zeros(len(seq_lengths_batch), dset.pad, dtype=torch.long)
        for i, seq_length in enumerate(seq_lengths_batch):
            assert seq_length % 3 == 0
            atom_id_list = internal_coords.BACKBONE_ATOM_IDS * (seq_length // 3)
            atom_ids_batch[i, :seq_length] = torch.tensor(atom_id_list, dtype=torch.long)
            atom_id_lists.append(atom_id_list)

        samples_batch = sample_batch(
            model=model,
            dset=dset,
            seq_lengths=seq_lengths_batch,
            atom_ids=atom_ids_batch,
            uniform=uniform,
            final_timepoint_only=final_timepoint_only,
            disable_pbar=disable_pbar,
        )
        samples.extend(samples_batch)

    # Label predictions
    if final_timepoint_only:
        samples = [pd.DataFrame(data=sample, columns=dset.feature_names) for sample in samples]
        for sample, atom_ids in zip(samples, atom_id_lists):
            sample["atom_label"] = [
                internal_coords.BACKBONE_ATOM_ID_TO_LABEL[atom_id] for atom_id in atom_ids
            ]

    return samples


def sample_unconditional(
    model: nn.Module,
    dset: noised.NoisedDataset,
    num_samples: int = 1,
    uniform: bool = False,
    batch_size: int = 65536,
    final_timepoint_only: bool = True,
    disable_pbar: bool = False,
) -> Union[List[np.ndarray], List[pd.DataFrame]]:
    """Sample num_samples samples by first sampling num_samples lengths using dset and then passing
    these to sample_unconditional_from_lengths."""
    seq_lengths = dset.sample_length(n=num_samples)
    if isinstance(seq_lengths, int):
        seq_lengths = [seq_lengths]
    return sample_unconditional_from_lengths(
        model,
        dset,
        seq_lengths,
        uniform=uniform,
        batch_size=batch_size,
        final_timepoint_only=final_timepoint_only,
        disable_pbar=disable_pbar,
    )


def sample_conditional(
    model: nn.Module,
    dset: noised.NoisedDataset,
    samples_multiplier: int = 2,
    samples_per_mol: Optional[int] = None,
    uniform: bool = False,
    batch_size: int = 65536,
    final_timepoint_only: bool = True,
    disable_pbar: bool = False,
) -> Union[
    Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, Union[List[int], List[np.ndarray]]]]
]:
    """Run reverse diffusion on a set of macrocycles conditioned on atom sequence.

    Args:
        model: Model.
        dset: Dataset to generate samples for (must contain atom features).
        samples_multiplier: For each molecule in the dataset, generate sample_multiplier * num_conformers samples.
        samples_per_mol: Override samples_multiplier to generate exactly this many samples per molecule.
        uniform: Sample uniformly instead of from a wrapped normal.
        batch_size: Batch size.
        final_timepoint_only: Only return the sample at the final (non-noisy) timepoint.
        disable_pbar: Don't display a progress bar.
    """
    if dset.atom_features is None:
        raise ValueError("Dataset must have atom features")

    num_samples_per_mol = []  # Per mol
    all_atom_idxs = []  # Per mol
    all_fnames = []  # Per mol
    seq_lengths = []  # Per conformer
    atom_features = []  # Per conformer

    for fname, structure in dset.structures.items():
        if samples_per_mol is None:
            num_conf = len(structure[dset.feature_names[0]])
            num_to_sample = samples_multiplier * num_conf
        else:
            num_to_sample = samples_per_mol

        atom_features_padded, atom_idxs = dset.get_atom_features(fname, pad=True, return_idxs=True)
        atom_features_padded_repeated = atom_features_padded.expand(num_to_sample, -1, -1)
        seq_length = len(atom_idxs)

        num_samples_per_mol.append(num_to_sample)
        all_atom_idxs.append(atom_idxs)
        all_fnames.append(fname)
        seq_lengths.extend(num_to_sample * [seq_length])
        atom_features.append(atom_features_padded_repeated)

    atom_features = torch.cat(atom_features)

    samples = []
    chunks = [(i, i + batch_size) for i in range(0, len(seq_lengths), batch_size)]

    logging.info(f"Sampling {len(seq_lengths)} items in batches of size {batch_size}")
    for idx_start, idx_end in chunks:
        samples_batch = sample_batch(
            model=model,
            dset=dset,
            seq_lengths=seq_lengths[idx_start:idx_end],
            atom_features=atom_features[idx_start:idx_end],
            uniform=uniform,
            final_timepoint_only=final_timepoint_only,
            disable_pbar=disable_pbar,
        )
        samples.extend(samples_batch)

    # samples is a flat list, need to map it back to mols
    mol_chunks = [0] + np.cumsum(num_samples_per_mol).tolist()
    samples_dict = {}

    # Aggregate samples for each molecule
    for mol_idx, (idx_start, idx_end) in enumerate(zip(mol_chunks, mol_chunks[1:])):
        # samples_mol is num_samples * [timesteps x seq_len x num_feat]
        # or num_samples * [seq_len x num_feat]
        samples_mol = samples[idx_start:idx_end]
        samples_mol = np.stack(samples_mol)  # [num_samples x ...]

        fname = all_fnames[mol_idx]
        structure = dset.structures[fname]
        atom_idxs = all_atom_idxs[mol_idx]

        if final_timepoint_only:  # Return as dataframes
            samples_mol_dict = {"atom_labels": structure["atom_labels"]}
            for feat_idx, feature_name in enumerate(dset.feature_names):
                df = pd.DataFrame(data=samples_mol[..., feat_idx], columns=atom_idxs)
                df.index.name = "sample_idx"
                feat_missing = structure[feature_name].iloc[0].isna()
                feat_missing_cols = feat_missing[feat_missing].index
                df[feat_missing_cols] = np.nan
                samples_mol_dict[feature_name] = df
            samples_dict[fname] = samples_mol_dict
        else:  # Return as arrays
            samples_mol_dict = {"atom_idxs": atom_idxs, "atom_labels": structure["atom_labels"]}
            for feat_idx, feature_name in enumerate(dset.feature_names):
                # [num_samples x timesteps x seq_len]
                samples_mol_dict[feature_name] = samples_mol[..., feat_idx]
            samples_dict[fname] = samples_mol_dict

    return samples_dict
