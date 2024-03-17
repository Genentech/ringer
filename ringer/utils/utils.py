import functools
import hashlib
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from typer.models import ParameterInfo


def get_wrapped_overlapping_sublists(list_: List[Any], size: int) -> Iterator[List[Any]]:
    for idx in range(len(list_)):
        idxs = [idx]
        for offset in range(1, size):
            # Wrap past end of list
            idxs.append((idx + offset) % len(list_))
        yield [list_[i] for i in idxs]


def get_overlapping_sublists(
    list_: List[Any], size: int, wrap: bool = True
) -> Iterator[List[Any]]:
    if wrap:
        for item in get_wrapped_overlapping_sublists(list_, size):
            yield item
    else:
        for i in range(len(list_) - size + 1):
            yield list_[i : i + size]


def compute_kl_divergence(p: np.ndarray, q: np.ndarray, nbins: int = 100) -> float:
    min_val = min(np.min(p), np.min(q))
    max_val = min(np.max(p), np.max(q))
    bins = np.linspace(min_val, max_val, nbins + 1)
    p_hist, _ = np.histogram(p, bins=bins)
    q_hist, _ = np.histogram(q, bins=bins)
    # Handle zero-counts
    p_hist[p_hist == 0] = 1
    q_hist[q_hist == 0] = 1
    return stats.entropy(p_hist, q_hist)


def compute_kl_divergence_from_dataframe(
    df: pd.DataFrame,
    *data_cols: str,
    key_col: str = "src",
    pkey: str = "Test",
    qkey: str = "Sampled",
    nbins: int = 100,
) -> pd.Series:
    dfp = df[df[key_col] == pkey]
    dfq = df[df[key_col] == qkey]
    return pd.Series(
        {col: compute_kl_divergence(dfp[col], dfq[col], nbins=nbins) for col in data_cols}
    )


def tolerant_comparison_check(values, cmp: Literal[">=", "<="], v):
    """Compares values in a way that is tolerant of numerical precision.

    >>> tolerant_comparison_check(-3.1415927410125732, ">=", -np.pi)
    True
    """
    if cmp == ">=":  # v is a lower bound
        minval = np.nanmin(values)
        diff = minval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True  # Passes
        return diff > 0
    elif cmp == "<=":
        maxval = np.nanmax(values)
        diff = maxval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True
        return diff < 0
    else:
        raise ValueError(f"Illegal comparator: {cmp}")


def modulo_with_wrapped_range(vals, range_min: float = -np.pi, range_max: float = np.pi):
    """Modulo with wrapped range -- capable of handing a range with a negative min.

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    return retval


def wrapped_mean(x: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    """Wrap the mean function about [-pi, pi]"""
    # https://rosettacode.org/wiki/Averages/Mean_angle
    sin_x = np.sin(x)
    cos_x = np.cos(x)

    retval = np.arctan2(np.nanmean(sin_x, axis=axis), np.nanmean(cos_x, axis=axis))
    return retval


def update_dict_nonnull(d: Dict[str, Any], vals: Dict[str, Any]) -> Dict[str, Any]:
    """Update a dictionary with values from another dictionary.

    >>> update_dict_nonnull({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    {'a': 1, 'b': 3, 'c': 4}
    """
    for k, v in vals.items():
        if k in d:
            if d[k] != v and v is not None:
                logging.info(f"Replacing key {k} original value {d[k]} with {v}")
                d[k] = v
        else:
            d[k] = v
    return d


def md5_all_py_files(dir_name: Union[str, Path]) -> str:
    """Create a single md5 sum for all given files."""
    # https://stackoverflow.com/questions/36099331/how-to-grab-all-files-in-a-folder-and-get-their-md5-hash-in-python
    dir_name = Path(dir_name)
    fnames = dir_name.glob("*.py")
    hash_md5 = hashlib.md5()
    for fname in sorted(fnames):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()


def unwrap_typer_args(func: Callable):
    # https://github.com/tiangolo/typer/issues/279#issuecomment-841875218
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        default_values = func.__defaults__
        patched_defaults = tuple(
            value.default if isinstance(value, ParameterInfo) else value
            for value in default_values
        )
        func.__defaults__ = patched_defaults

        return func(*args, **kwargs)

    return wrapper
