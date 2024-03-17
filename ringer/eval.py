#!/usr/bin/env python

import functools
import json
import logging
import multiprocessing
import pickle
import subprocess
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pandas as pd
import torch
import typer
from rdkit import Chem

from ringer.data import macrocycle, noised
from ringer.models import bert_for_diffusion
from ringer.utils import (
    data_loading,
    internal_coords,
    peptides,
    plotting,
    sampling,
    utils,
)


class Split(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"

    def __str__(self) -> str:
        return self.value


def stack_structures(
    structures: Dict[str, Dict[str, Any]],
    feature_names: Sequence[str],
    num_conf: Optional[int] = None,
) -> pd.DataFrame:
    dfs = []
    for structure in structures.values():
        # If num_conf is None, uses all conformers
        long_data = {}
        actual_num_conf = 0
        for name in feature_names:
            feats = structure[name].iloc[:num_conf]
            actual_num_conf = len(feats)
            feats_flat = feats.to_numpy().ravel()
            long_data[name] = feats_flat
        df = pd.DataFrame(data=long_data)
        df["atom_label"] = structure["atom_labels"] * actual_num_conf
        df["num_residues"] = f"{len(structure['atom_labels']) // 3} residues"
        dfs.append(df)
    return pd.concat(dfs)


def get_side_chain_feat_names(mol: Chem.Mol) -> Dict[int, Tuple[str, ...]]:
    side_chain_torsion_idxs = peptides.get_side_chain_torsion_idxs(mol)
    side_chain_feat_names = {}
    feat_names = macrocycle.MacrocycleAnglesWithSideChainsDataset.side_chain_feature_names

    for backbone_atom_idx, chain_atom_idxs in side_chain_torsion_idxs.items():
        angle_idxs = chain_atom_idxs[1:]
        angle_idxs = list(utils.get_overlapping_sublists(angle_idxs, 3, wrap=False))
        dihedral_idxs = chain_atom_idxs
        dihedral_idxs = list(utils.get_overlapping_sublists(dihedral_idxs, 4, wrap=False))

        side_chain_feat_names[backbone_atom_idx] = feat_names["angle"][: len(angle_idxs)]
        side_chain_feat_names[backbone_atom_idx] += feat_names["dihedral"][: len(dihedral_idxs)]

    # Mapping from backbone index to feature names for the side chain attached to it
    return side_chain_feat_names


def stack_structures_side_chains(
    structures: Dict[str, Dict[str, Any]], num_conf: Optional[int] = None
) -> pd.DataFrame:
    data = defaultdict(list)

    for path, structure in structures.items():
        # Need to only extract feature names that actually exist
        with open(path, "rb") as f:
            mol = pickle.load(f)["rd_mol"]
        feat_name_dict = get_side_chain_feat_names(mol)

        for backbone_atom_idx, feat_names in feat_name_dict.items():
            for feat_name in feat_names:
                feats = structure[feat_name][backbone_atom_idx].iloc[:num_conf]
                data[feat_name].extend(feats)

    data = {k: data[k] for k in sorted(data)}
    dfs = []
    for k, v in data.items():
        df = pd.DataFrame({"value": v})
        df["feature"] = k
        dfs.append(df)
    df = pd.concat(dfs)

    return df


def label_and_combine_side_chain_data(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    dfs = []
    for dataset_label, dataframe in dataframes.items():
        dataframe["src"] = dataset_label
        dfs.append(dataframe)
    combined_data = pd.concat(dfs)
    return combined_data


def label_and_combine_data(dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # We know the backbone atom labels, so we can assign angle and dihedral labels accordingly
    angle_labels = ["theta_1", "theta_2", "theta_3"]
    dihedral_labels = ["phi", "psi", "omega"]
    get_angle_label = dict(zip(internal_coords.BACKBONE_ATOM_LABELS, angle_labels))
    get_dihedral_label = dict(zip(internal_coords.BACKBONE_ATOM_LABELS, dihedral_labels))

    dfs = []
    for dataset_label, dataframe in dataframes.items():
        dataframe["angle_label"] = dataframe["atom_label"].map(get_angle_label)
        dataframe["dihedral_label"] = dataframe["atom_label"].map(get_dihedral_label)
        dataframe["src"] = dataset_label
        dfs.append(dataframe)
    combined_data = pd.concat(dfs)

    return combined_data


def get_phi_psi_data(combined_data: pd.DataFrame) -> pd.DataFrame:
    # Make dataframe with separate columns for phi and psi to make Ramachandran plot
    phi = combined_data[combined_data["dihedral_label"] == "phi"][
        ["src", "num_residues", "dihedral"]
    ]
    psi = combined_data[combined_data["dihedral_label"] == "psi"][["dihedral"]]
    phi = phi.reset_index(drop=True).rename(columns={"dihedral": "phi"})
    psi = psi.reset_index(drop=True).rename(columns={"dihedral": "psi"})
    phi_psi_data = pd.concat([phi, psi], axis=1)
    return phi_psi_data


def unconditional(
    samples: List[pd.DataFrame],
    dset: noised.NoisedDataset,
    out_dir: Union[str, Path],
    num_conf: Optional[int] = None,
    ext: str = ".png",
) -> Dict[str, pd.DataFrame]:
    """Perform evaluation of unconditional model, i.e., plot distributions and Ramachandran plots.

    Args:
        samples: List of samples from unconditional model.
        dset: Dataset containing reference data.
        out_dir: Directory to write output/plots to.
        num_conf: Number of conformers to use from reference data.
        ext: File extension for plots.

    Returns:
        Stacked samples and reference data.
    """
    out_dir = Path(out_dir)

    # Stack data
    logging.info("Formatting data")
    test_data = stack_structures(dset.structures, dset.feature_names, num_conf=num_conf)
    for sample in samples:
        sample["num_residues"] = f"{len(sample) // 3} residues"
    sampled_data = pd.concat(samples)

    dataframes = {"Sampled": sampled_data, "Test": test_data}
    combined_data = label_and_combine_data(dataframes)

    logging.info("Plotting")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plotting.plot_angle_and_dihedral_distributions(combined_data, plot_dir, ext=ext)
    plotting.plot_angle_and_dihedral_distributions(combined_data, plot_dir, ext=ext, residues=True)

    phi_psi_data = get_phi_psi_data(combined_data)
    num_test = len(phi_psi_data[phi_psi_data["src"] == "Test"])
    num_sampled = len(phi_psi_data[phi_psi_data["src"] == "Sampled"])
    # Make num_sampled <= num_test
    phi_psi_data_1x = phi_psi_data.groupby("src").sample(n=min(num_test, num_sampled))
    plotting.plot_ramachandran_plots(phi_psi_data, plot_dir, as_rows=True)
    plotting.plot_ramachandran_plots(phi_psi_data, plot_dir, residues=True)
    plotting.plot_ramachandran_plots(
        phi_psi_data_1x, plot_dir, as_rows=True, name="ramachandran_1x"
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_1x, plot_dir, name="ramachandran_1x", residues=True
    )

    return dataframes


def save_mols(mol_dir: Union[str, Path], mols: Dict[str, Chem.Mol]) -> None:
    mol_dir = Path(mol_dir)
    mol_dir.mkdir(exist_ok=True)

    for fname, mol in mols.items():
        mol_path = mol_dir / Path(fname).name
        with open(mol_path, "wb") as sink:
            pickle.dump(mol, sink)


def load_mols(
    mol_dir: Union[str, Path], fnames: Iterable[Union[str, Path]]
) -> Dict[str, Chem.Mol]:
    mol_dir = Path(mol_dir)

    mols = {}
    for fname in fnames:
        mol_path = mol_dir / Path(fname).name
        with open(mol_path, "rb") as source:
            mol = pickle.load(source)
        mols[fname] = mol

    return mols


def reconstruct_mols(
    out_dir: Union[str, Path],
    mol_dir: Union[str, Path],
    mean_distances_path: Union[str, Path],
    mean_angles_path: Optional[Union[str, Path]] = None,
    std_angles_path: Optional[Union[str, Path]] = None,
    skip_opt: bool = False,
    max_conf: Optional[int] = None,
    num_proc: int = multiprocessing.cpu_count(),
) -> subprocess.CompletedProcess:
    reconstruct_script_path = Path(__file__).resolve().parent / "reconstruct.py"
    samples_path = Path(out_dir) / "samples.pickle"
    cmd = [
        "python",
        str(reconstruct_script_path),
        str(mol_dir),
        str(samples_path),
        str(out_dir),
        str(mean_distances_path),
        "--save-unsuccessful",
        "--ncpu",
        str(num_proc),
    ]
    if mean_angles_path is not None and std_angles_path is not None:
        cmd.extend(
            [
                "--mean-angles-path",
                str(mean_angles_path),
                "--std-angles-path",
                str(std_angles_path),
            ]
        )
    if skip_opt:
        cmd.append("--skip-opt")
    if max_conf is not None:
        cmd.extend(["--max-conf", str(max_conf)])
    completed_process = subprocess.run(cmd)
    if completed_process.returncode != 0:
        logging.warning(
            f"Running '{reconstruct_script_path.name}' completed with exit status {completed_process.returncode}"
        )
    return completed_process


def compute_metrics(
    out_dir: Union[str, Path],
    mol_dir: Union[str, Path],
    mol_opt_dir: Union[str, Path],
    num_proc: int = multiprocessing.cpu_count(),
) -> subprocess.CompletedProcess:
    compute_metrics_script_path = Path(__file__).resolve().parent / "compute_metrics.py"
    cmd = [
        "python",
        str(compute_metrics_script_path),
        str(mol_dir),
        str(mol_opt_dir),
        str(out_dir),
        "--ncpu",
        str(num_proc),
    ]
    completed_process = subprocess.run(cmd)
    if completed_process.returncode != 0:
        logging.warning(
            f"Running '{compute_metrics_script_path.name}' completed with exit status {completed_process.returncode}"
        )
    return completed_process


def get_coverage_dataframe(metrics: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    rmsd_cov = metrics["ring-rmsd"]["cov"]
    tfd_cov = metrics["ring-tfd"]["cov"]
    rmsd_cov = rmsd_cov.reset_index(level="threshold").melt(
        id_vars="threshold", var_name="cov-type", value_name="cov"
    )
    tfd_cov = tfd_cov.reset_index(level="threshold").melt(
        id_vars="threshold", var_name="cov-type", value_name="cov"
    )
    rmsd_cov["src"] = "RMSD"
    tfd_cov["src"] = "TFD"
    cov = pd.concat([rmsd_cov, tfd_cov])
    cov["cov-type"] = cov["cov-type"].map(str.upper)  # Upper case
    cov["cov"] *= 100  # Convert to %
    return cov


def conditional(
    samples: Dict[str, Dict[str, pd.DataFrame]],
    dset: noised.NoisedDataset,
    out_dir: Union[str, Path],
    mean_distances_path: Union[str, Path],
    mean_angles_path: Optional[Union[str, Path]] = None,
    std_angles_path: Optional[Union[str, Path]] = None,
    do_reconstruct: bool = True,
    do_metrics: bool = True,
    skip_opt: bool = False,
    max_opt_conf: Optional[int] = None,
    num_conf: Optional[int] = None,
    ext: str = ".png",
    num_proc: int = multiprocessing.cpu_count(),
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    """Perform evaluation of conditional model, i.e., plot distributions and Ramachandran plots,
    perform optimization to recover consistent macrocycle angles, and compute metrics.

    Args:
        samples: Samples from conditional model.
        dset: Dataset containing reference data.
        out_dir: Directory to write output/plots to.
        mean_distances_path: Path to file containing mean distances from training data.
        mean_angles_path: Path to file containing mean bond angles from training data.
        std_angles_path: Path to file containing bond angle stdevs from training data.
        do_reconstruct: Whether to reconstruct mols, reconstructed mols must already be present in out_dir if False.
        do_metrics: Whether to compute metrics, metrics must already be present in out_dir if False.
        max_opt_conf: Reconstruct at most this many conformers.
        num_conf: Number of conformers to use from reference data.
        ext: File extension for plots.
        num_proc: Number of parallel workers.

    Returns:
        Stacked samples, reference data, and reconstructed data.
    """
    out_dir = Path(out_dir)

    # Stack data
    logging.info("Formatting test data")
    test_data = stack_structures(dset.structures, dset.feature_names, num_conf=num_conf)
    sampled_data = stack_structures(samples, dset.feature_names)

    dataframes = {"Sampled": sampled_data, "Test": test_data}
    combined_data = label_and_combine_data(dataframes)

    logging.info("Plotting")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plotting.plot_angle_and_dihedral_distributions(combined_data, plot_dir, ext=ext)
    plotting.plot_angle_and_dihedral_distributions(combined_data, plot_dir, ext=ext, residues=True)

    phi_psi_data = get_phi_psi_data(combined_data)
    num_test = len(phi_psi_data[phi_psi_data["src"] == "Test"])
    num_sampled = len(phi_psi_data[phi_psi_data["src"] == "Sampled"])
    # Make num_sampled <= num_test
    phi_psi_data_1x = phi_psi_data.groupby("src").sample(n=min(num_test, num_sampled))
    plotting.plot_ramachandran_plots(phi_psi_data, plot_dir, as_rows=True)
    plotting.plot_ramachandran_plots(phi_psi_data, plot_dir, residues=True)
    plotting.plot_ramachandran_plots(
        phi_psi_data_1x, plot_dir, as_rows=True, name="ramachandran_1x"
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_1x, plot_dir, name="ramachandran_1x", residues=True
    )

    # Reconstruct/optimize mols
    mol_dir = out_dir / "mols"
    mol_opt_dir = out_dir / "reconstructed_mols"
    mols = {fname: dset.atom_features[fname]["mol"] for fname in samples.keys()}
    save_mols(mol_dir, mols)

    # Reconstruct in separate Python call to avoid multiprocessing overhead
    if do_reconstruct:
        logging.info(f"Reconstructing rings with {num_proc} processes")
        reconstruct_mols(
            out_dir=out_dir,
            mol_dir=mol_dir,
            mean_distances_path=mean_distances_path,
            mean_angles_path=mean_angles_path,
            std_angles_path=std_angles_path,
            skip_opt=skip_opt,
            max_conf=max_opt_conf,
            num_proc=num_proc,
        )
    elif any(not (mol_opt_dir / Path(fname).name).exists() for fname in mols.keys()):
        raise ValueError(f"Missing reconstructed mols in '{mol_opt_dir}'")

    samples_reconstructed_path = out_dir / "samples_reconstructed.pickle"
    with open(samples_reconstructed_path, "rb") as source:
        samples_reconstructed = pickle.load(source)

    # Make plots with reconstructed data
    logging.info("Plotting")
    reconstructed_data = stack_structures(samples_reconstructed, dset.feature_names)
    dataframes["Reconstructed"] = reconstructed_data
    combined_data_with_reconstructed = label_and_combine_data(dataframes)
    phi_psi_data_with_reconstructed = get_phi_psi_data(combined_data_with_reconstructed)
    num_test = len(
        phi_psi_data_with_reconstructed[phi_psi_data_with_reconstructed["src"] == "Test"]
    )
    num_sampled = len(
        phi_psi_data_with_reconstructed[phi_psi_data_with_reconstructed["src"] == "Sampled"]
    )
    # Make num_sampled <= num_test
    phi_psi_data_with_reconstructed_1x = phi_psi_data_with_reconstructed.groupby("src").sample(
        n=min(num_test, num_sampled)
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed,
        plot_dir,
        name="ramachandran_with_reconstructed",
        col_order=["Test", "Sampled", "Reconstructed"],
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed,
        plot_dir,
        name="ramachandran_with_reconstructed",
        col_order=["Test", "Sampled", "Reconstructed"],
        residues=True,
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed_1x,
        plot_dir,
        name="ramachandran_with_reconstructed_1x",
        col_order=["Test", "Sampled", "Reconstructed"],
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed_1x,
        plot_dir,
        name="ramachandran_with_reconstructed_1x",
        col_order=["Test", "Sampled", "Reconstructed"],
        residues=True,
    )

    # Compute metrics
    if do_metrics:
        logging.info(f"Computing metrics with {num_proc} processes")
        compute_metrics(
            out_dir=out_dir, mol_dir=mol_dir, mol_opt_dir=mol_opt_dir, num_proc=num_proc
        )

    metrics_path = out_dir / "metrics.pickle"
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    metrics_agg_path = out_dir / "metrics_aggregated.pickle"
    with open(metrics_agg_path, "rb") as f:
        metrics_agg = pickle.load(f)

    coverage = get_coverage_dataframe(metrics)
    coverage_path = plot_dir / f"coverage{ext}"
    plotting.plot_coverage(coverage, path=coverage_path)

    return dataframes, metrics_agg


def conditional_side_chains(
    samples: Dict[str, Dict[str, pd.DataFrame]],
    dset: noised.NoisedDataset,
    out_dir: Union[str, Path],
    mean_distances_path: Union[str, Path],
    mean_angles_path: Optional[Union[str, Path]] = None,
    std_angles_path: Optional[Union[str, Path]] = None,
    do_reconstruct: bool = True,
    do_metrics: bool = True,
    skip_opt: bool = False,
    max_opt_conf: Optional[int] = None,
    num_conf: Optional[int] = None,
    ext: str = ".png",
    num_proc: int = multiprocessing.cpu_count(),
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    """Perform evaluation of conditional model with side chains, i.e., plot distributions and
    Ramachandran plots, perform optimization to recover consistent macrocycle angles, and compute
    metrics.

    Args:
        samples: Samples from conditional model.
        dset: Dataset containing reference data.
        out_dir: Directory to write output/plots to.
        mean_distances_path: Path to file containing mean distances from training data.
        mean_angles_path: Path to file containing mean bond angles from training data.
        std_angles_path: Path to file containing bond angle stdevs from training data.
        do_reconstruct: Whether to reconstruct mols, reconstructed mols must already be present in out_dir if False.
        do_metrics: Whether to compute metrics, metrics must already be present in out_dir if False.
        max_opt_conf: Reconstruct at most this many conformers.
        num_conf: Number of conformers to use from reference data.
        ext: File extension for plots.
        num_proc: Number of parallel workers.

    Returns:
        Stacked samples, reference data, and reconstructed data.
    """
    out_dir = Path(out_dir)

    # Stack data
    logging.info("Formatting test data")
    test_data = stack_structures(dset.structures, ["angle", "dihedral"], num_conf=num_conf)
    sampled_data = stack_structures(samples, ["angle", "dihedral"])

    dataframes = {"Sampled": sampled_data, "Test": test_data}
    combined_data = label_and_combine_data(dataframes)

    test_data_sc = stack_structures_side_chains(dset.structures, num_conf=num_conf)
    sampled_data_sc = stack_structures_side_chains(samples, num_conf=num_conf)

    dataframes_sc = {"Sampled": sampled_data_sc, "Test": test_data_sc}
    combined_data_sc = label_and_combine_side_chain_data(dataframes_sc)

    logging.info("Plotting")
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plotting.plot_angle_and_dihedral_distributions(combined_data, plot_dir, ext=ext)
    plotting.plot_angle_and_dihedral_distributions(combined_data, plot_dir, ext=ext, residues=True)
    plotting.plot_side_chain_distributions(combined_data_sc, plot_dir=plot_dir, ext=ext)

    phi_psi_data = get_phi_psi_data(combined_data)
    num_test = len(phi_psi_data[phi_psi_data["src"] == "Test"])
    num_sampled = len(phi_psi_data[phi_psi_data["src"] == "Sampled"])
    # Make num_sampled <= num_test
    phi_psi_data_1x = phi_psi_data.groupby("src").sample(n=min(num_test, num_sampled))
    plotting.plot_ramachandran_plots(phi_psi_data, plot_dir, as_rows=True)
    plotting.plot_ramachandran_plots(phi_psi_data, plot_dir, residues=True)
    plotting.plot_ramachandran_plots(
        phi_psi_data_1x, plot_dir, as_rows=True, name="ramachandran_1x"
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_1x, plot_dir, name="ramachandran_1x", residues=True
    )

    # Reconstruct/optimize mols
    mol_dir = out_dir / "mols"
    mol_opt_dir = out_dir / "reconstructed_mols"
    mols = {fname: dset.atom_features[fname]["mol"] for fname in samples.keys()}
    save_mols(mol_dir, mols)

    # Reconstruct in separate Python call to avoid multiprocessing overhead
    if do_reconstruct:
        logging.info(f"Reconstructing rings with {num_proc} processes")
        reconstruct_mols(
            out_dir=out_dir,
            mol_dir=mol_dir,
            mean_distances_path=mean_distances_path,
            mean_angles_path=mean_angles_path,
            std_angles_path=std_angles_path,
            skip_opt=skip_opt,
            max_conf=max_opt_conf,
            num_proc=num_proc,
        )
    elif any(not (mol_opt_dir / Path(fname).name).exists() for fname in mols.keys()):
        raise ValueError(f"Missing reconstructed mols in '{mol_opt_dir}'")

    samples_reconstructed_path = out_dir / "samples_reconstructed.pickle"
    with open(samples_reconstructed_path, "rb") as source:
        samples_reconstructed = pickle.load(source)

    # Make plots with reconstructed data
    logging.info("Plotting")
    reconstructed_data = stack_structures(samples_reconstructed, ["angle", "dihedral"])
    dataframes["Reconstructed"] = reconstructed_data
    combined_data_with_reconstructed = label_and_combine_data(dataframes)
    phi_psi_data_with_reconstructed = get_phi_psi_data(combined_data_with_reconstructed)
    num_test = len(
        phi_psi_data_with_reconstructed[phi_psi_data_with_reconstructed["src"] == "Test"]
    )
    num_sampled = len(
        phi_psi_data_with_reconstructed[phi_psi_data_with_reconstructed["src"] == "Sampled"]
    )
    # Make num_sampled <= num_test
    phi_psi_data_with_reconstructed_1x = phi_psi_data_with_reconstructed.groupby("src").sample(
        n=min(num_test, num_sampled)
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed,
        plot_dir,
        name="ramachandran_with_reconstructed",
        col_order=["Test", "Sampled", "Reconstructed"],
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed,
        plot_dir,
        name="ramachandran_with_reconstructed",
        col_order=["Test", "Sampled", "Reconstructed"],
        residues=True,
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed_1x,
        plot_dir,
        name="ramachandran_with_reconstructed_1x",
        col_order=["Test", "Sampled", "Reconstructed"],
    )
    plotting.plot_ramachandran_plots(
        phi_psi_data_with_reconstructed_1x,
        plot_dir,
        name="ramachandran_with_reconstructed_1x",
        col_order=["Test", "Sampled", "Reconstructed"],
        residues=True,
    )

    # Compute metrics
    if do_metrics:
        logging.info(f"Computing metrics with {num_proc} processes")
        compute_metrics(
            out_dir=out_dir, mol_dir=mol_dir, mol_opt_dir=mol_opt_dir, num_proc=num_proc
        )

    metrics_path = out_dir / "metrics.pickle"
    with open(metrics_path, "rb") as f:
        metrics = pickle.load(f)

    metrics_agg_path = out_dir / "metrics_aggregated.pickle"
    with open(metrics_agg_path, "rb") as f:
        metrics_agg = pickle.load(f)

    coverage = get_coverage_dataframe(metrics)
    coverage_path = plot_dir / f"coverage{ext}"
    plotting.plot_coverage(coverage, path=coverage_path)

    return dataframes, metrics_agg


@utils.unwrap_typer_args
def evaluate(
    # Model/output
    model_dir: str = typer.Option(
        "results", help="Model directory", rich_help_panel="Model and Output"
    ),
    out_dir: str = typer.Option(
        "sample",
        help="Directory to write samples, plots, and evaluation results to",
        rich_help_panel="Model and Output",
    ),
    # Data
    data_dir: Optional[str] = typer.Option(
        None,
        help=f"Data directory containing pickle files, can be relative to '{Path(__file__).resolve().parent.parent / 'data'}' (inferred from trainings args if None)",
        rich_help_panel="Data Loading",
    ),
    use_data_cache: bool = typer.Option(
        True, help="Use/build data cache", rich_help_panel="Data Loading"
    ),
    data_cache_dir: Optional[str] = typer.Option(
        None, help="Directory to save/load data cache to/from", rich_help_panel="Data Loading"
    ),
    unsafe_cache: bool = typer.Option(
        False,
        help="Don't check data filenames and cache hashes before loading data",
        rich_help_panel="Data Loading",
    ),
    split: Split = typer.Option(
        Split.test, help="Data split to load", rich_help_panel="Data Loading"
    ),
    split_sizes: Tuple[float, float, float] = typer.Option(
        (0.8, 0.1, 0.1), help="Split sizes for splitting data", rich_help_panel="Data Loading"
    ),
    # Sampling
    num_samples: int = typer.Option(
        2,
        help="For unconditional: generate num_samples * num_all_confs; for conditional: generate num_samples * num_conformers samples for each molecule in the dataset",
        rich_help_panel="Sampling",
    ),
    as_multiplier: bool = typer.Option(
        True,
        help="For unconditional: if no-as-multiplier, generate num_samples total samples; for conditional: if no-as-multiplier, generate num_samples samples for each molecule in the dataset",
        rich_help_panel="Sampling",
    ),
    num_conf: Optional[int] = typer.Option(
        None,
        help="Number of conformers to use from test set for plotting and evaluation",
        rich_help_panel="Sampling",
    ),
    uniform: bool = typer.Option(
        False,
        help="Sample uniformly instead of from wrapped normal",
        rich_help_panel="Sampling",
    ),
    batch_size: int = typer.Option(
        65536, help="Batch size for sampling", rich_help_panel="Sampling"
    ),
    seed: int = typer.Option(
        42,
        help="Random seed for sampling (doesn't affect data splitting)",
        rich_help_panel="Sampling",
    ),
    sample_only: bool = typer.Option(
        False, help="Skip evaluation and only perform sampling", rich_help_panel="Sampling"
    ),
    # Other
    skip_opt: bool = typer.Option(
        False,
        help="Skip optimization during reconstruction and just set Cartesian coordinates by going through the sequence linearly",
        rich_help_panel="Other",
    ),
    max_opt_conf: Optional[int] = typer.Option(
        None,
        help="Maximum number of conformers to reconstruct",
        rich_help_panel="Other",
    ),
    eval_only: bool = typer.Option(
        False,
        help="Skip sampling and only perform evaluation (reconstruction, metrics, plotting), 'samples.pickle' must be present in outdir",
        rich_help_panel="Other",
    ),
    metrics_only: bool = typer.Option(
        False,
        help="Skip sampling and reconstruction and only compute metrics, 'samples.pickle', 'samples_reconstructed.pickle', and mol dirs must be present in outdir",
        rich_help_panel="Other",
    ),
    plotting_only: bool = typer.Option(
        False,
        help="Skip sampling, reconstruction, computing metrics and only make plots, 'metrics.pickle' and 'metrics_aggregated.pickle must be present in outdir",
        rich_help_panel="Other",
    ),
    ext: str = typer.Option("png", help="File extension for plots", rich_help_panel="Other"),
    device: str = typer.Option(
        "cuda", help="Device to use for inference", rich_help_panel="Other"
    ),
    ncpu: int = typer.Option(
        multiprocessing.cpu_count(), help="Number of workers", rich_help_panel="Other"
    ),
    overwrite: bool = typer.Option(
        False, help="Overwrite output directory", rich_help_panel="Other"
    ),
) -> Optional[
    Union[
        Dict[str, pd.DataFrame], Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]
    ]
]:
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise ValueError(f"Model directory '{model_dir}' doesn't exist")

    if ext.startswith("."):
        raise ValueError("Specify extension without leading period")
    ext = f".{ext}"

    train_args_path = model_dir / "training_args.json"
    train_means_path = model_dir / "training_mean_offset.json"
    train_mean_distances_path = model_dir / "training_mean_distances.json"
    train_mean_angles_path = model_dir / "training_mean_angles.json"
    train_std_angles_path = model_dir / "training_std_angles.json"
    with open(train_args_path, "r") as source:
        train_args = json.load(source)
    with open(train_means_path, "r") as source:
        train_means = json.load(source)  # Need these to offset by *training* means
    with open(train_mean_distances_path, "r") as source:
        train_mean_distances = json.load(source)  # Need these to reconstruct Cartesians
    if train_mean_angles_path.exists():
        with open(train_mean_angles_path, "r") as source:
            train_mean_angles = json.load(source)
    else:
        train_mean_angles_path = None
        train_mean_angles = None
    if train_std_angles_path.exists():
        with open(train_std_angles_path, "r") as source:
            train_std_angles = json.load(source)
    else:
        train_std_angles_path = None
        train_std_angles = None

    if data_dir is None:
        data_dir = train_args["data_dir"]

    dsets = data_loading.get_datasets(
        data_dir=data_dir,
        internal_coordinates_definitions=train_args["internal_coordinates_definitions"],
        splits=[split],
        split_sizes=split_sizes,
        use_atom_features=train_args["use_atom_features"],
        atom_feature_fingerprint_radius=train_args["atom_feature_fingerprint_radius"],
        atom_feature_fingerprint_size=train_args["atom_feature_fingerprint_size"],
        max_conf="all",  # Always use all confs for evaluation
        timesteps=train_args["timesteps"],
        variance_schedule=train_args["variance_schedule"],
        variance_scale=train_args["variance_scale"],
        use_cache=use_data_cache,
        cache_dir=data_cache_dir,
        unsafe_cache=unsafe_cache,
        num_proc=ncpu,
        sample_seed=seed,
    )
    dset = dsets[split]

    # Update means to training set means
    dset.means = train_means
    logging.info(f"Setting means to training data means {dset.means}")

    # Update mean distances to training set mean distances
    logging.info(f"Setting mean distances to training data mean distances {train_mean_distances}")
    dset.dset.atom_type_means["distance"] = train_mean_distances
    if train_mean_angles is not None:
        logging.info(
            f"Setting mean bond angles to training data mean bond angles {train_mean_angles}"
        )
        dset.dset.atom_type_means["angle"] = train_mean_angles
    if train_std_angles is not None:
        logging.info(
            f"Setting bond angle stdevs to training data bond angle stdevs {train_std_angles}"
        )
        dset.dset.atom_type_stdevs["angle"] = train_std_angles

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=eval_only or overwrite)

    if plotting_only or metrics_only:
        logging.info("Disabling sampling")
        eval_only = True
    if train_args["use_atom_features"]:  # Conditional generation
        sample_func = functools.partial(
            sampling.sample_conditional,
            samples_multiplier=num_samples,
            samples_per_mol=None if as_multiplier else num_samples,
        )
        do_reconstruct = do_metrics = True
        if metrics_only:
            logging.info("Disabling reconstruction")
            do_reconstruct = False
        if plotting_only:
            logging.info("Disabling reconstruction and computing metrics")
            do_reconstruct = do_metrics = False
        eval_func = functools.partial(
            conditional_side_chains
            if isinstance(dset.dset, macrocycle.MacrocycleAnglesWithSideChainsDataset)
            else conditional,
            mean_distances_path=train_mean_distances_path,
            mean_angles_path=train_mean_angles_path,
            std_angles_path=train_std_angles_path,
            do_reconstruct=do_reconstruct,
            do_metrics=do_metrics,
            skip_opt=skip_opt,
            max_opt_conf=max_opt_conf,
            num_proc=ncpu,
        )
    else:  # Unconditional generation
        if as_multiplier:
            num_samples = num_samples * len(dset)
        sample_func = functools.partial(
            sampling.sample_unconditional,
            num_samples=num_samples,
        )
        eval_func = unconditional

    samples_path = out_dir / "samples.pickle"
    if eval_only:  # Load previously generated samples
        logging.info("Loading samples")
        with open(samples_path, "rb") as source:
            samples = pickle.load(source)
    else:  # Perform sampling
        model_snapshot_dir = out_dir / "model_snapshot"

        logging.info(f"Loading model from '{model_dir}'")
        model = (
            bert_for_diffusion.BertForDiffusion.from_dir(model_dir, copy_to=model_snapshot_dir)
            .to(torch.device(device))
            .eval()
        )

        torch.manual_seed(seed)  # Set for reproducibility
        samples = sample_func(model=model, dset=dset, batch_size=batch_size, uniform=uniform)
        with open(samples_path, "wb") as sink:
            pickle.dump(samples, sink)

    # Perform evaluation
    if not sample_only:
        return eval_func(samples, dset, out_dir, num_conf=num_conf, ext=ext)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    typer.run(evaluate)


if __name__ == "__main__":
    main()
