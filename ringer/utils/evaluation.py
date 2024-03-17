import multiprocessing
from collections import defaultdict
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from . import chem, internal_coords


def compute_cov_mat_metrics(
    confusion_mat: np.ndarray,
    thresholds: np.ndarray = np.arange(0.05, 1.55, 0.05),
) -> Dict[str, Union[Dict[str, float], Dict[str, np.ndarray]]]:
    # confusion_mat: num_ref x num_gen
    ref_min = confusion_mat.min(axis=1)
    gen_min = confusion_mat.min(axis=0)
    cov_thresh = ref_min.reshape(-1, 1) <= thresholds.reshape(1, -1)
    jnk_thresh = gen_min.reshape(-1, 1) <= thresholds.reshape(1, -1)

    cov_r = cov_thresh.mean(axis=0)
    mat_r = ref_min.mean()
    cov_p = jnk_thresh.mean(axis=0)
    mat_p = gen_min.mean()

    cov = {"threshold": thresholds, "cov-r": cov_r, "cov-p": cov_p}
    mat = {"mat-r": mat_r, "mat-p": mat_p}

    return {"cov": cov, "mat": mat}


def get_atom_map(probe_mol: Chem.Mol, ref_mol: Chem.Mol) -> List[Dict[int, int]]:
    ref_mol_idxs = [a.GetIdx() for a in ref_mol.GetAtoms()]
    matches = probe_mol.GetSubstructMatches(
        ref_mol, uniquify=False
    )  # Don't uniquify to account for symmetry
    atom_map = [dict(zip(match, ref_mol_idxs)) for match in matches]
    return atom_map


def compute_rmsd_matrix(
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    conf_ids_probe: Optional[List[int]] = None,
    conf_ids_ref: Optional[List[int]] = None,
    ncpu: int = 1,
) -> np.ndarray:
    probe_mol = Chem.RemoveHs(probe_mol)
    ref_mol = Chem.RemoveHs(ref_mol)

    # Precompute atom map for faster RMSD calculation
    atom_map = get_atom_map(probe_mol, ref_mol)
    atom_map = [list(map_.items()) for map_ in atom_map]

    rmsd_mat = compute_rmsd_matrix_from_map(
        probe_mol,
        ref_mol,
        atom_map,
        conf_ids_probe=conf_ids_probe,
        conf_ids_ref=conf_ids_ref,
        ncpu=ncpu,
    )
    return rmsd_mat  # [num_ref x num_probe]


def compute_ring_rmsd_matrix(
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    conf_ids_probe: Optional[List[int]] = None,
    conf_ids_ref: Optional[List[int]] = None,
    ncpu: int = 1,
) -> np.ndarray:
    probe_mol = Chem.RemoveHs(probe_mol)
    ref_mol = Chem.RemoveHs(ref_mol)

    # Precompute atom map for faster RMSD calculation
    atom_map = get_atom_map(probe_mol, ref_mol)

    # Subset to macrocycle indices
    probe_macrocycle_idxs = chem.get_macrocycle_idxs(probe_mol, n_to_c=False)
    if probe_macrocycle_idxs is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(probe_mol)}'"
        )
    atom_map = list(set(tuple((k, map_[k]) for k in probe_macrocycle_idxs) for map_ in atom_map))

    # Check to make sure we have all macrocycle indices in the ref mol
    ref_macrocycle_idxs = chem.get_macrocycle_idxs(ref_mol, n_to_c=False)
    if ref_macrocycle_idxs is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(ref_mol)}'"
        )
    ref_macrocycle_idxs = set(ref_macrocycle_idxs)
    for map_ in atom_map:
        if set(ref_idx for _, ref_idx in map_) != ref_macrocycle_idxs:
            raise ValueError("Inconsistent macrocycle indices")

    rmsd_mat = compute_rmsd_matrix_from_map(
        probe_mol,
        ref_mol,
        atom_map,
        conf_ids_probe=conf_ids_probe,
        conf_ids_ref=conf_ids_ref,
        ncpu=ncpu,
    )
    return rmsd_mat  # [num_ref x num_probe]


def compute_rmsd_matrix_from_map(
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    atom_map: List[Sequence[Tuple[int, int]]],
    conf_ids_probe: Optional[List[int]] = None,
    conf_ids_ref: Optional[List[int]] = None,
    ncpu: int = 1,
) -> np.ndarray:
    if conf_ids_ref is None:
        conf_ids_ref = [conf.GetId() for conf in ref_mol.GetConformers()]
    if conf_ids_probe is None:
        conf_ids_probe = [conf.GetId() for conf in probe_mol.GetConformers()]

    num_ref = len(conf_ids_ref)
    num_probe = len(conf_ids_probe)

    _get_best_rms_args = []
    for i, ref_id in enumerate(conf_ids_ref):
        for j, probe_id in enumerate(conf_ids_probe):
            _get_best_rms_args.append((i, j, probe_id, ref_id))

    pfunc = partial(_get_best_rms, probe_mol=probe_mol, ref_mol=ref_mol, atom_map=atom_map)
    with multiprocessing.Pool(ncpu) as pool:
        rmsds_with_idxs = pool.map(pfunc, _get_best_rms_args)

    rmsd_mat = np.empty((num_ref, num_probe))
    for i, j, rmsd in rmsds_with_idxs:
        rmsd_mat[i, j] = rmsd

    return rmsd_mat  # [num_ref x num_probe]


def _get_best_rms(
    args: Tuple[int, int, int, int],
    probe_mol: Chem.Mol,
    ref_mol: Chem.Mol,
    atom_map: List[Sequence[Tuple[int, int]]],
) -> Tuple[int, int, float]:
    i, j, probe_id, ref_id = args
    rmsd = AllChem.GetBestRMS(probe_mol, ref_mol, prbId=probe_id, refId=ref_id, map=atom_map)
    return i, j, rmsd


def compute_ring_tfd_matrix(probe_mol: Chem.Mol, ref_mol: Chem.Mol, ncpu: int = 1) -> np.ndarray:
    """Compute ring torsion fingerprint deviation as in
    https://doi.org/10.1021/acs.jcim.0c00025."""
    probe_mol = Chem.RemoveHs(probe_mol)
    ref_mol = Chem.RemoveHs(ref_mol)

    # Precompute atom map for faster RMSD calculation
    ref_mol_idxs = [a.GetIdx() for a in ref_mol.GetAtoms()]
    matches = probe_mol.GetSubstructMatches(
        ref_mol, uniquify=False
    )  # Don't uniquify to account for symmetry
    atom_map = [dict(zip(match, ref_mol_idxs)) for match in matches]

    # Subset to macrocycle indices
    probe_macrocycle_idxs = chem.get_macrocycle_idxs(probe_mol, n_to_c=False)
    if probe_macrocycle_idxs is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(probe_mol)}'"
        )
    atom_map = list(set(tuple((k, map_[k]) for k in probe_macrocycle_idxs) for map_ in atom_map))

    num_torsions = len(probe_macrocycle_idxs)
    num_conf_probe = probe_mol.GetNumConformers()
    num_conf_ref = ref_mol.GetNumConformers()

    probe_torsions = internal_coords.get_macrocycle_dihedrals(probe_mol, probe_macrocycle_idxs)
    probe_torsions = probe_torsions.to_numpy()
    probe_torsions_tiled = np.tile(probe_torsions[np.newaxis, ...], (num_conf_ref, 1, 1))

    ref_macrocycle_idxs_check = chem.get_macrocycle_idxs(ref_mol, n_to_c=False)
    if ref_macrocycle_idxs_check is None:
        raise ValueError(
            f"Macrocycle indices could not be determined for '{Chem.MolToSmiles(ref_mol)}'"
        )
    ref_macrocycle_idxs_check = set(ref_macrocycle_idxs_check)

    ref_macrocycle_idxs_list = []
    for map_ in atom_map:
        ref_macrocycle_idxs = [ref_idx for _, ref_idx in map_]  # Same order as in probe
        if set(ref_macrocycle_idxs) != ref_macrocycle_idxs_check:
            raise ValueError("Inconsistent macrocycle indices")
        ref_macrocycle_idxs_list.append(ref_macrocycle_idxs)

    # There could be multiple maps, so select the minimum TFD for each one
    pfunc = partial(
        _get_ring_tfd_for_ref_idxs,
        ref_mol=ref_mol,
        num_conf_probe=num_conf_probe,
        probe_torsions_tiled=probe_torsions_tiled,
        num_torsions=num_torsions,
    )
    with multiprocessing.Pool(ncpu) as pool:
        tfds = pool.map(pfunc, ref_macrocycle_idxs_list)
    tfd = np.minimum.reduce(tfds)

    return tfd  # [num_ref x num_probe]


def _get_ring_tfd_for_ref_idxs(
    ref_macrocycle_idxs: List[int],
    ref_mol: Chem.Mol,
    num_conf_probe: int,
    probe_torsions_tiled: np.ndarray,
    num_torsions: int,
) -> np.ndarray:
    ref_torsions = internal_coords.get_macrocycle_dihedrals(ref_mol, ref_macrocycle_idxs)
    ref_torsions = ref_torsions.to_numpy()

    # Compute deviations between all pairs of conformers
    ref_torsions_tiled = np.tile(ref_torsions[:, np.newaxis, :], (1, num_conf_probe, 1))
    torsion_deviation = probe_torsions_tiled - ref_torsions_tiled

    # Wrap deviation around [-pi, pi] range
    torsion_deviation = (torsion_deviation + np.pi) % (2 * np.pi) - np.pi

    # Scale by max deviation, sum across torsions, and normalize
    tfd = np.sum(np.abs(torsion_deviation) / np.pi, axis=-1) / num_torsions

    return tfd


class CovMatEvaluator:
    confusion_mat_funcs = {
        "rmsd": compute_rmsd_matrix,  # heavy atoms
        "ring-rmsd": compute_ring_rmsd_matrix,
        "ring-tfd": compute_ring_tfd_matrix,
    }
    thresholds = {
        "rmsd": np.arange(0, 2.51, 0.01),
        "ring-rmsd": np.arange(0, 1.26, 0.01),
        "ring-tfd": np.arange(0, 1.01, 0.01),  # Can't be larger than 1
    }

    def __init__(self, metrics: Sequence[str] = ("ring-rmsd", "ring-tfd")) -> None:
        for name in metrics:
            if name not in self.confusion_mat_funcs:
                raise NotImplementedError(f"Metric '{name}' is not implemented")
        self.metric_names = metrics

    def __call__(
        self, probe_mol: Chem.Mol, ref_mol: Chem.Mol, ncpu: int = 1
    ) -> Dict[str, Dict[str, Union[Dict[str, float], Dict[str, np.ndarray]]]]:
        metrics = {}
        for metric_name in self.metric_names:
            confusion_mat_func = self.confusion_mat_funcs[metric_name]
            confusion_mat = confusion_mat_func(probe_mol, ref_mol, ncpu=ncpu)
            metric = compute_cov_mat_metrics(
                confusion_mat, thresholds=self.thresholds[metric_name]
            )
            metrics[metric_name] = metric
        return metrics

    @staticmethod
    def stack_results(
        metrics_dict: Dict[
            str, Dict[str, Dict[str, Union[Dict[str, float], Dict[str, np.ndarray]]]]
        ]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        # Stack all COV/MAT for each metric
        cov = defaultdict(dict)
        mat = defaultdict(list)

        for fname, results in metrics_dict.items():
            # results = {'metric1': {'cov': ..., 'mat': ...}, ...}
            for metric_name, result in results.items():
                # result = {'cov': {'threshold': ..., 'cov-r': ..., 'cov-p': ...}, 'mat': {'mat-r': ..., 'mat-p': ...}}
                cov[metric_name][fname] = pd.DataFrame(result["cov"]).set_index("threshold")
                mat[metric_name].append(pd.DataFrame(result["mat"], index=[fname]))

        cov = {metric_name: pd.concat(results) for metric_name, results in cov.items()}
        mat = {metric_name: pd.concat(results) for metric_name, results in mat.items()}

        # {'metric1': {'cov': pd.DataFrame, 'mat': pd.DataFrame}, ...}
        results = {
            metric_name: {"cov": cov[metric_name], "mat": mat[metric_name]}
            for metric_name in cov.keys()
        }

        return results

    @staticmethod
    def aggregate_results_for_metric(results: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return {
            "cov": results["cov"].groupby(level="threshold").agg(func=["mean", "median"]),
            "mat": results["mat"].agg(func=["mean", "median"]),
        }

    @classmethod
    def aggregate_results(
        cls, results: Dict[str, Dict[str, pd.DataFrame]]
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        return {
            metric_name: cls.aggregate_results_for_metric(r) for metric_name, r in results.items()
        }
