from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from matplotlib.offsetbox import AnchoredText
from scipy import interpolate

from . import utils


def plot_ramachandran(
    data: pd.DataFrame,
    x: str = "phi",
    y: str = "psi",
    col: Optional[str] = None,
    col_wrap: Optional[int] = None,
    col_order: Optional[List[str]] = None,
    row: Optional[str] = None,
    row_order: Optional[List[str]] = None,
    height: int = 4,
    plot_type: Literal["density_scatter", "hexbin", "sns.scatterplot"] = "density_scatter",
    remove_axis_text: bool = False,
    path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> sns.FacetGrid:
    g = sns.FacetGrid(
        data,
        col=col,
        col_wrap=col_wrap,
        col_order=col_order,
        row=row,
        row_order=row_order,
        sharex=True,
        sharey=True,
        height=height,
        aspect=1,
        despine=False,
        margin_titles=row is not None,
        gridspec_kws=dict(wspace=0.05, hspace=0.05),
    )
    name, *attrs = plot_type.split(".")
    plot_func = globals()[name]
    for attr in attrs:
        plot_func = getattr(plot_func, attr)
    g.map(plot_func, x, y, **kwargs)
    format_facet_grid_dihedral_axis(g, which="x")
    format_facet_grid_dihedral_axis(g, which="y")
    g.set_xlabels(r"$\phi$")
    g.set_ylabels(r"$\psi$")
    g.tick_params(axis="both", labelsize=8)
    if row is None:
        g.set_titles(template="{col_name}")
    else:
        g.set_titles(col_template="{col_name}", row_template="{row_name}")
    for ax in g.axes.flat:
        ax.set_aspect("equal")
        if remove_axis_text:
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
    if path is not None:
        g.figure.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    return g


def plot_distributions(
    data: pd.DataFrame,
    x: str,
    col: Optional[str] = None,
    col_wrap: Optional[int] = None,
    row: Optional[str] = None,
    hue: Optional[str] = None,
    binwidth: float = 2 * np.pi / 100,
    sharex: bool = True,
    sharey: bool = False,
    height: int = 3,
    format_as_dihedral_axis: bool = True,
    xlabel: Optional[str] = None,
    latex: bool = True,
    path: Optional[Union[str, Path]] = None,
    add_kl_div: bool = True,
    **kwargs,
) -> sns.FacetGrid:
    g = sns.FacetGrid(
        data,
        hue=hue,
        col=col,
        col_wrap=col_wrap,
        row=row,
        sharex=sharex,
        sharey=sharey,
        height=height,
        margin_titles=row is not None,
        **kwargs,
    )
    g.map(sns.histplot, x, stat="density", binwidth=binwidth, linewidth=0.2)
    if row is None:
        g.set_titles("")
    else:
        g.set_titles(col_template="", row_template="{row_name}")
    if format_as_dihedral_axis:
        format_facet_grid_dihedral_axis(g)
    if xlabel is None:
        for label, ax in g.axes_dict.items():
            if row is None:
                if latex:
                    ax.set_xlabel(rf"$\{label}$")
                else:
                    ax.set_xlabel(label)
            else:
                if latex:
                    ax.set_xlabel(rf"$\{label[1]}$")
                else:
                    ax.set_xlabel(label[1])
    else:
        g.set_xlabels(label=xlabel)
    g.set_ylabels(label="Density")
    g.tick_params(axis="y", labelsize=8)
    g.add_legend(title="", loc="upper left")
    if add_kl_div:

        def add_kldiv_to_plot(ax, kl_div):
            at = AnchoredText(
                f"KL = {kl_div:.4f}", prop=dict(size=10), frameon=True, loc="upper right"
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            at.patch.set(alpha=0.8, edgecolor=(0.8, 0.8, 0.8, 0.8))
            ax.add_artist(at)

        if col is None and row is None:
            kl_div = utils.compute_kl_divergence_from_dataframe(data, x)
            add_kldiv_to_plot(g.ax, kl_div[x])
        else:
            by = list(filter(None, [row, col]))
            kl_divs = data.groupby(by).apply(
                lambda d: utils.compute_kl_divergence_from_dataframe(d, x)
            )
            for label, ax in g.axes_dict.items():
                if kl_divs is not None:
                    kl_div = kl_divs.loc[label, x]
                    add_kldiv_to_plot(ax, kl_div)
    g.figure.tight_layout(pad=0.01, h_pad=0.1, w_pad=0.1)
    if path is not None:
        g.figure.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    return g


def plot_coverage(
    data: pd.DataFrame,
    x: str = "threshold",
    y: str = "cov",
    hue: str = "cov-type",
    col: str = "src",
    col_order: Sequence[str] = ("RMSD", "TFD"),
    path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> sns.FacetGrid:
    g = sns.FacetGrid(
        data,
        hue=hue,
        col=col,
        col_order=col_order,
        sharex=False,
        height=4,
        despine=False,
        xlim=(0, data[x].max()),
        ylim=(0, 100),
        legend_out=False,
        **kwargs,
    )
    g.map(sns.lineplot, x, y)
    xlabels = {"RMSD": "Threshold (Å)", "TFD": "Threshold"}
    for label, ax in g.axes_dict.items():
        ax.set_xlabel(xlabels[label])
        if label == "TFD":
            ax.set_xlim(0, 1)
    g.set_ylabels("Coverage (%)")
    g.set_titles(template="{col_name}")
    g.add_legend(title="")
    if path is not None:
        g.figure.savefig(path, dpi=600, bbox_inches="tight", pad_inches=0.01)
    return g


def plot_ramachandran_plots(
    phi_psi_data: pd.DataFrame,
    plot_dir: Optional[Union[str, Path]] = None,
    name: str = "ramachandran",
    ext: str = ".png",
    col_order: Sequence[str] = ("Test", "Sampled"),
    as_rows: bool = False,
    residues: bool = False,
) -> None:
    if plot_dir is not None:
        plot_dir = Path(plot_dir)
        fname = f"{name}_residues{ext}" if residues else f"{name}{ext}"
        ramachandran_path = plot_dir / fname
    else:
        ramachandran_path = None

    col = "src"
    col_order = list(col_order)
    row = "num_residues" if residues else None
    row_order = ["4 residues", "5 residues", "6 residues"] if residues else None
    if as_rows:
        row, col = col, row
        row_order, col_order = col_order, row_order

    plot_ramachandran(
        phi_psi_data,
        col=col,
        col_order=col_order,
        row=row,
        row_order=row_order,
        height=3,
        s=1,
        edgecolors="none",
        cmap="magma",
        path=ramachandran_path,
    )


def plot_angle_and_dihedral_distributions(
    data: pd.DataFrame,
    plot_dir: Optional[Union[str, Path]] = None,
    ext: str = ".png",
    residues: bool = False,
) -> None:
    def get_path(name: str) -> Optional[Path]:
        if plot_dir is None:
            return None
        if residues:
            name += "_residues"
        fname = f"{name}{ext}"
        return Path(plot_dir) / fname

    if "angle" in data.columns:
        plot_distributions(
            data=data,
            x="angle",
            row="num_residues" if residues else None,
            row_order=["4 residues", "5 residues", "6 residues"] if residues else None,
            hue="src",
            binwidth=np.pi / 200,
            format_as_dihedral_axis=False,
            xlim=(1.5, 2.5),
            xlabel="Bond angle",
            height=2 if residues else 3,
            aspect=1.5,
            legend_out=False,
            despine=False,
            path=get_path("angle_dist"),
        )
        plot_distributions(
            data=data,
            x="angle",
            row="num_residues" if residues else None,
            row_order=["4 residues", "5 residues", "6 residues"] if residues else None,
            col="angle_label",
            hue="src",
            height=2 if residues else 3,
            aspect=1.4 if residues else 1,
            binwidth=np.pi / 200,
            format_as_dihedral_axis=False,
            xlim=(1.5, 2.5),
            legend_out=False,
            despine=False,
            path=get_path("angles_dists"),
        )

    if "dihedral" in data.columns:
        plot_distributions(
            data=data,
            x="dihedral",
            row="num_residues" if residues else None,
            row_order=["4 residues", "5 residues", "6 residues"] if residues else None,
            hue="src",
            binwidth=2 * np.pi / 60,
            xlabel="Dihedral angle",
            height=2 if residues else 3,
            aspect=1.5,
            legend_out=False,
            despine=False,
            path=get_path("dihedral_dist"),
        )
        plot_distributions(
            data=data,
            x="dihedral",
            row="num_residues" if residues else None,
            row_order=["4 residues", "5 residues", "6 residues"] if residues else None,
            col="dihedral_label",
            hue="src",
            binwidth=2 * np.pi / 60,
            height=2 if residues else 3,
            aspect=1.4 if residues else 1,
            legend_out=False,
            despine=False,
            path=get_path("dihedrals_dists"),
        )


def plot_side_chain_distributions(
    data: pd.DataFrame,
    plot_dir: Optional[Union[str, Path]] = None,
    ext: str = ".png",
) -> None:
    def get_path(name: str) -> Optional[Path]:
        if plot_dir is None:
            return None
        fname = f"{name}{ext}"
        return Path(plot_dir) / fname

    plot_distributions(
        data=data[data["feature"].str.startswith("sc_a")],
        x="value",
        col="feature",
        col_wrap=5,
        hue="src",
        height=2,
        aspect=1,
        binwidth=np.pi / 200,
        format_as_dihedral_axis=False,
        xlabel="value",
        latex=False,
        xlim=(1.5, 2.5),
        legend_out=False,
        despine=False,
        path=get_path("sidechain_angle_dists"),
    )
    plot_distributions(
        data=data[data["feature"].str.startswith("sc_chi")],
        x="value",
        col="feature",
        col_wrap=5,
        hue="src",
        height=2,
        aspect=1,
        binwidth=2 * np.pi / 60,
        format_as_dihedral_axis=True,
        latex=False,
        legend_out=False,
        despine=False,
        path=get_path("sidechain_dihedral_dists"),
    )


def format_facet_grid_dihedral_axis(g: sns.FacetGrid, which: Literal["x", "y"] = "x") -> None:
    lim = (-np.pi, np.pi)
    ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    ticklabels = [r"-$\pi$", r"-$\pi/2$", 0, r"$\pi/2$", r"$\pi$"]
    g.set(**{f"{which}lim": lim, f"{which}ticks": ticks, f"{which}ticklabels": ticklabels})


def hexbin(
    x: np.ndarray,
    y: np.ndarray,
    color: Optional[Union[str, Tuple[float, float, float]]] = None,
    gridsize: int = 50,
    bins: Union[Literal["log"], int, Sequence[float]] = "log",
    **kwargs,
):
    cmap = kwargs.pop("cmap", None)
    if cmap is None and color is not None:
        cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=gridsize, bins=bins, cmap=cmap, **kwargs)


def density_scatter(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    ax: Optional[mpl.axes.Axes] = None,
    bins: Tuple[int, int] = (500, 500),
    color: Optional[Union[str, Tuple[float, float, float]]] = None,
    norm: ImageNormalize = ImageNormalize(vmin=0, vmax=1, stretch=LogStretch()),
    **kwargs,
) -> mpl.axes.Axes:
    # https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib
    if ax is None:
        ax = plt.gca()

    x = np.asarray(x)
    y = np.asarray(y)

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, density=True)
    points = (0.5 * (xedges[1:] + xedges[:-1]), 0.5 * (yedges[1:] + yedges[:-1]))
    points, hist = augment_with_periodic_bc(points, hist, domain=(2 * np.pi, 2 * np.pi))
    z = interpolate.interpn(
        points,
        hist,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
        fill_value=0,
    )

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    cmap = kwargs.pop("cmap", None)
    if cmap is None and color is not None:
        cmap = sns.light_palette(color, as_cmap=True)

    ax.scatter(x, y, c=z, cmap=cmap, norm=norm, **kwargs)
    return ax


def augment_with_periodic_bc(
    points: Tuple[np.ndarray, ...],
    values: np.ndarray,
    domain: Optional[Union[float, Sequence[float]]] = None,
) -> Tuple[Tuple[np.ndarray, ...], np.ndarray]:
    """Augment the data to create periodic boundary conditions.

    Parameters
    ----------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions.
    domain : float or None or array_like of shape (n, )
        The size of the domain along each of the n dimensions
        or a uniform domain size along all dimensions if a
        scalar. Using None specifies aperiodic boundary conditions.

    Returns
    -------
    points : tuple of ndarray of float, with shapes (m1, ), ..., (mn, )
        The points defining the regular grid in n dimensions with
        periodic boundary conditions.
    values : array_like, shape (m1, ..., mn, ...)
        The data on the regular grid in n dimensions with periodic
        boundary conditions.
    """
    # https://stackoverflow.com/questions/25087111/2d-interpolation-with-periodic-boundary-conditions
    # Validate the domain argument
    n = len(points)
    if np.ndim(domain) == 0:
        domain = [domain] * n
    if np.shape(domain) != (n,):
        raise ValueError("`domain` must be a scalar or have the same " "length as `points`")

    # Pre- and append repeated points
    points = [
        x if d is None else np.concatenate([x - d, x, x + d]) for x, d in zip(points, domain)
    ]

    # Tile the values as necessary
    reps = [1 if d is None else 3 for d in domain]
    values = np.tile(values, reps)

    return points, values
