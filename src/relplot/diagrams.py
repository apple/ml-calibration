#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from functools import partial
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from deprecation import deprecated

from matplotlib.collections import LineCollection
from sklearn.ensemble import BaggingRegressor

from . import config
from .estimators import Binning, KernelSmoother, CenteredRegressor
from .kernels import TruncatedGaussianKernel, ReflectedGaussianKernel
from .metrics import smECE_fast, binnedECE, _get_default_kernel


def plot_line(x, y, colors, ax, **kwargs):
    # See: https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    points = np.array([x, y]).T.reshape((-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments, colors=colors, linestyle="solid", joinstyle="round", **kwargs
    )
    line = ax.add_collection(lc)
    return line


def set_default_style():
    mpl.rc_file_defaults()
    sns.set_style("whitegrid")
    pal = sns.color_palette("pastel")
    sns.set_palette(pal, color_codes=True)

    mpl.rcParams.update(
        {
            "axes.edgecolor": '0.5',
            "font.size": 22,
            "legend.frameon": False,
            "patch.force_edgecolor": False,
            "figure.figsize": [6.0, 6.0],
            "axes.titlepad": 20,
        }
    )

    tex_fonts = {
        "font.family": "serif",
        "text.usetex": True,
        "text.latex.preamble": r"""
            \usepackage{libertine}
            \usepackage[libertine]{newtxmath}
            """,
    }
    if config.use_tex_fonts:
        mpl.rcParams.update(tex_fonts)


def reliability_diagram(predictions, labels, estimator, centered_fit=True):
    labels = labels.reshape(-1)
    f = predictions.reshape((-1, 1))
    y = labels
    if centered_fit:
        estimator = CenteredRegressor(estimator)
    estimator.fit(f, y)

    t = np.linspace(0, 1, 1000)
    mu = estimator.predict(t.reshape(-1, 1)).reshape(-1)
    return t, mu


def compute_split_densities(f, y, x_eval=None, sigma=0.1, density_kernel = ReflectedGaussianKernel):
    # Computes one density for f|y=0, one for f|y=1
    ker = density_kernel(sigma)
    if x_eval is None:
        x_eval = np.linspace(0, 1, 100)

    outputs = []
    for fi in f[y == 0], f[y == 1]:
        density = ker.kde(fi, x_eval)
        outputs.append(density)
    return outputs


def prepare_rel_diagram_binned(f, y, nbins=15):
    f, y = map(lambda x: np.array(x, dtype=np.double).reshape(-1).copy(), [f, y])
    binning = Binning(nbins)
    t, mu = reliability_diagram(f, y, binning, centered_fit=False)

    buckets = binning.bucket_predictions_.copy()
    alphas = np.array(
        [bs / np.sum(binning.bucket_sizes_) for bs in binning.bucket_sizes_]
    )
    alphas /= np.max(alphas)
    ece = binnedECE(f, y, nbins)
    return {'t': t, 'mu' : mu, 'buckets': buckets, 'alphas': alphas, 'ece' : ece }

def plot_rel_diagram_binned(t, mu, buckets, alphas, ece, fig=None, ax=None):
    set_default_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    nbins = len(buckets)
    for i, y_bucket in enumerate(buckets):
        yb = y_bucket
        lb, ub = i / nbins, (i + 1) / nbins
        ax.stairs([yb], [lb, ub], fill=True, color="gray", alpha=alphas[i])
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect('equal')

    ax.plot(t, t, 'k--', lw=1, alpha=0.3)
    ax.plot(t, mu, 'r-')

    #ax.set_aspect("equal")

    ece_label = (
        f"$\\mathrm{{ECE}}_{{{nbins}}}: {ece:.3f}$"
        if config.use_tex_fonts
        else f"ECE$_{{{nbins}}}: {ece:.3f}$"
    )
    ax.text(0.05, 0.9, ece_label)
    if config.use_tex_fonts:
        ax.set_xlabel("$f$")
        ax.set_ylabel(r"$\mathbb{E}[ y \mid f ]$")
    else:
        ax.set_xlabel("f")
        ax.set_ylabel(r"E[ y | f ]")
    return fig, ax

def rel_diagram_binned(f, y, nbins=15, fig=None, ax=None):
    diagram = prepare_rel_diagram_binned(f, y, nbins=nbins)
    return plot_rel_diagram_binned(**diagram, fig=fig, ax=ax)


def prepare_rel_diagram(
    f : npt.ArrayLike,
    y : npt.ArrayLike,
    plot_confidence_band=True,
    plot_bag_lines=False,
    num_bootstrap=200,
    kde_bandwidth=None,
    report_CE=True,
    report_CE_std=True,
    custom_regressor=None,
    kernel = None,
    **unused_kwargs,
):
    """Computes calibration data for plotting reliability diagrams.
    Args:
        f (npt.ArrayLike): Array of predictions f(x_i) in [0, 1].
        y (npt.ArrayLike): Array of binary observations (ground-truth labels) y_i in {0, 1}.
        fig (matplotlib.figure.Figure, optional): Figure to use, if provided. Defaults to None.
        plot_confidence_band (bool, optional): Compute bootstrapped confidence bands around the main line. Defaults to True.
        plot_bag_lines (bool, optional): Compute the individual bootstrapped estimators. Defaults to False.
        num_bootstrap (int, optional): Number of bootstrap estimators. Defaults to 200.
        kde_bandwidth (float, optional): Override the default choice of bandwidth for plotting densities, if specified. Defaults to None.
        report_CE (bool, optional): Print the calibration error (smECE) on the plot. Defaults to True.
        report_CE_std (bool, optional): Compute and print a 95% confidence interval of the calibration error, estimated via bootstrapping. Defaults to True.
        custom_regressor (sklearn.base.BaseEstimator, optional): Use a custom sklearn estimator as the regressor, if specified. Defaults to None.
        kernel (optional): Use a custom kernel for smoothing and density estimation, if specified. Defaults to kernels.ReflectedGaussianKernel.

    Returns:
        Dictionary containing keys:
            - "mu" : The main reliability curve, evaluated on the mesh (np.ndarray).
            - "mesh": np.ndarray of grid points on which densities and functions are evaluated
            - "ce" : The computed calibration error (smECE).
            - "ce_ci_width": Width of 95% confidence interval on calibration error (bootstrap estimate).
            - "density": Kernel density estimates of the distribution of predictions f
            - "densities": Returns returns a list of two kernel density estimates of f, conditioned on y=0 and y=1 respectively.
            - "upper" : Upper bound of confidence band on "mu".
            - "lower" : Lower bound of confidence band on "mu".

    """
    if kernel is None:
        kernel = ReflectedGaussianKernel

    def predict(regr, x):
        return regr.predict(x.reshape(-1, 1)).reshape(-1)

    f, y = map(lambda x: np.array(x, dtype=np.double).reshape(-1).copy(), [f, y])

    ## Setup the mesh
    outputs = {}
    t = np.linspace(0, 1, config.plot_mesh_pts)
    outputs["mesh"] = t

    ## Compute the smECE
    ice, opt_width = smECE_fast(f, y, eps=0.001, return_width=True)

    ## Compute the kde
    sigma = opt_width if kde_bandwidth is None else kde_bandwidth
    outputs["sigma"] = sigma
    density = kernel(sigma).kde(f, t)
    outputs["density"] = density


    if not custom_regressor:
        estimator = KernelSmoother(kernel(sigma))
    else:
        estimator = custom_regressor

    ## Bootstrapped regressions
    outputs["bags"] = []
    if plot_bag_lines or plot_confidence_band:
        bag = BaggingRegressor(estimator=estimator, n_estimators=num_bootstrap).fit(
                f.reshape(-1, 1), y
                )
        dfs = []
        for i, est in enumerate(bag.estimators_):
            mu = predict(est, t)
            dfs.append(pd.DataFrame.from_records({"x": t, "ey": mu}))

            if plot_bag_lines and i < 50:
                outputs["bags"].append(mu)
        mu = predict(bag, t)
    else:
        estimator.fit(f.reshape(-1, 1), y)
        mu = predict(estimator, t)

    ## The main line
    outputs["mu"] = mu

    if plot_confidence_band:
        df = pd.concat(dfs, ignore_index=True)
        qwidth = 0.95
        lower = df.groupby("x").quantile((1 - qwidth) / 2)["ey"].to_numpy()
        upper = df.groupby("x").quantile(1 - (1 - qwidth) / 2)["ey"].to_numpy()
        outputs["lower"] = lower
        outputs["upper"] = upper

    outputs["densities"] = compute_split_densities(f , y, x_eval=t, sigma=sigma, density_kernel=kernel)

    ## The endpoint densities
    eps = 0.005 # intervals [0, eps] and [1-eps, 1] get collapsed into 'endpoints'
    for x0 in [0, 1]:
        mesh = t
        I = np.where( np.abs(mesh - x0) <= eps )
        outputs[f'{x0}pt_mu'] = mu[I].mean()
        outputs[f'{x0}pt_density'] = density[I].sum() / density.sum()
        if plot_confidence_band:
            outputs[f'{x0}pt_lower'] = lower[I].mean()
            outputs[f'{x0}pt_upper'] = upper[I].mean()

    if report_CE:
        outputs["ce"] = ice
        if report_CE_std:
            res = scipy.stats.bootstrap(
                (f, y),
                smECE_fast,
                paired=True,
                n_resamples=100,
                method="basic",
                confidence_level=0.95,
            )
            ci = res.confidence_interval
            wid = max(ice - ci.low, ci.high - ice)
            outputs["ce_ci_width"] = wid

    ## Include sub-samples
    I = np.random.default_rng(seed=42).permutation(len(f))[:100] # at most 100 samples
    outputs['f_samp'] = f[I]
    outputs['y_samp'] = y[I]

    return outputs


def plot_rel_diagram(
    diagram,
    fig=None,
    ax=None,
    use_default_style=True,
    plot_main_line=True,
    simple_main_line=False,
    plot_density=False,
    plot_density_ticks=True,
    split_densities=False,
    color='red',
    density_color='gray',
    special_endpoints=False,
    endpoint_prob_thresh=0.01,
    plot_labels=True,
    plot_diagonal=True,
    **unused_kwargs,
):
    """
    Args:
        diagram (dict): Calibration data; result of relplot.prepare_rel_diagram
        fig (matplotlib.figure.Figure, optional): Figure to use, if provided. Defaults to None.
        ax (matplotlib.axes.Axes, optional): Axes to use, if provided. Defaults to None.
        use_default_style (bool, optional): Apply the recommended plot styling. Defaults to True.
        color (str, optional): Color to use for the main regression line. Defaults to 'red'.
        density_color (str, optional): Color to use for the density overlays. Defaults to 'gray'.
        plot_main_line (bool, optional): Plot the primary reliability curve, the regression function. Defaults to True.
        plot_confidence_band (bool, optional): Plot bootstrapped confidence bands around the main line. Defaults to True.
        plot_bag_lines (bool, optional): Plot the individual bootstrapped estimators. Defaults to False.
        num_bootstrap (int, optional): Number of bootstrap estimators. Defaults to 200.
        split_densities (bool, optional): Plot separate densities for f(x) on positive and negative labels, instead of a single density. Defaults to False.
        refined_kde (bool, optional): Use cross-validation to pick the kernel bandwidth for plotting the densities. Defaults to False.
        kde_bandwidth (float, optional): Override the default choice of bandwidth for plotting densities, if specified. Defaults to None.
        report_CE (bool, optional): Print the calibration error (smECE) on the plot. Defaults to True.
        report_CE_std (bool, optional): Compute and print a 95% confidence interval of the calibration error, estimated via bootstrapping. Defaults to True.
        custom_regressor (sklearn.base.BaseEstimator, optional): Use a custom sklearn estimator as the regressor, if specified. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Reliability diagram figure.
    """
    
    def get_ticks_from_density(dens, mesh, n_ticks=200):
        return np.random.default_rng(seed=0).choice(mesh, size=n_ticks, p=dens/np.sum(dens))
    
    def get_sizes_from_density(density):
        SIZE_SCALE = 200
        density /= 2
        filt = density < 1
        density_n = (density**2) * filt + np.sqrt(density) * (1-filt)
        return (density_n) * SIZE_SCALE


    if use_default_style:
        set_default_style()

    color = np.array(mpl.colors.to_rgba(color))

    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=(6, 6))
 

    ## main diagram
    t = diagram["mesh"]
    if plot_diagonal:
        ax.plot(t, t, "k--", lw=1, alpha=0.2)

    density = diagram["density"]
    density_n = density / np.max(density)

    ## bootstrapped lines (plot max 50 of them)
    for mu in diagram["bags"]:
        alpha_scale = 0.20
        plot_line(
            t, mu, ax=ax, colors=[np.clip(color * di * alpha_scale, 0, 1) for di in density], lw=1
        )

    ## main red line (simple)
    if plot_main_line and simple_main_line:
        mu = diagram["mu"]
        # colors = [color * np.array([1, 1, 1, np.clip(di + 0.1, 0, 1)]) for di in density]
        # plot_line(t, mu, colors=colors, ax=ax, lw=2, ls="--") 
        ax.plot(t, mu, '--', color=color, clip_on=False)

    ## main red line (density-dependent width)
    if plot_main_line and not simple_main_line:
        mesh_d = np.linspace(0, 1, 1000) # denser mesh for the line
        mu = diagram["mu"]
        mu_d = np.interp(mesh_d, diagram['mesh'], diagram['mu']) # denser mu
        density_d = np.interp(mesh_d, diagram['mesh'], diagram['density'])
        sizes = get_sizes_from_density(density_d)
        colors = [color * np.array([1, 1, 1, np.clip(di * 0.3, 0, 1)]) for di in density_d]
        ax.scatter(mesh_d, mu_d, marker='.', c=colors, s=sizes, clip_on=False, zorder=100)
        # linewidths = density_d * 50
        # plot_line(mesh_d, mu_d, ax=ax, colors=colors, linewidths=linewidths, clip_on=False, zorder=100)


    ## confidence bands
    if "upper" in diagram.keys():
        upper = diagram["upper"]
        lower = diagram["lower"]
        for i in range(len(t) - 1):
            ax.fill_between(
                [t[i], t[i + 1]],
                [lower[i], lower[i + 1]],
                [upper[i], upper[i + 1]],
                lw=0,
                edgecolor=None,
                # alpha = np.clip(density[i] * 0.5, 0, 1),
                alpha = density_n[i] * 0.6,
                color='gray',
                clip_on=True
            )


    ## endpoint densities
    if special_endpoints:
        for x0 in [0, 1]:
            mu = diagram[f'{x0}pt_mu']
            pt_dens = diagram[f'{x0}pt_density']
            if pt_dens < endpoint_prob_thresh:
                continue # Don't plot small probabilities

            msize = 100 * (pt_dens)
            lw = msize / 3
            ax.plot([x0], [mu], color=color, clip_on=False, zorder=100, marker='o', markersize=msize,markerfacecolor=color, markeredgecolor='none', alpha=1)
            if f'{x0}pt_upper' in diagram.keys():
                ub = diagram[f'{x0}pt_upper']
                lb = diagram[f'{x0}pt_lower']
                _, _, lines = ax.errorbar(x0, mu, yerr=[[mu-lb], [ub-mu]], clip_on=False, zorder=100, capsize=0, marker='none', lw=lw, color=color) # capsize=3
                lines[0].set_capstyle('round')

            ha = 'left' if x0 == 0 else 'right'
            xform = ax.transData.inverted()
            width_pt = (xform.transform((msize, 0)) - xform.transform((0, 0)))[0] / 2.0
            text_shift = (0.01 + width_pt ) * (1 if x0 == 0 else -1)
            ax.text(x=x0 + text_shift, y=mu, s=f'({pt_dens:.2f})', fontsize='x-small', va='center', ha=ha, clip_on=False)

    if plot_labels:
        if config.use_tex_fonts:
            ax.set_xlabel("$f$")
            ax.set_ylabel(r"$\mathbb{E}[ y \mid f ]$")
        else:
            ax.set_xlabel("f")
            ax.set_ylabel(r"E[ y | f ]")


    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_aspect('equal')

    ## density subplot
    if plot_density:
        densities = diagram["densities"] if split_densities else [diagram["density"]]
        alpha = 0.3
        for i, dens in enumerate(densities):
            if not split_densities:
                hatch = None
            else:
                hatch = "//" if i == 0 else "\\\\"
            max_density_height = 0.3
            height = dens / np.max(dens) * max_density_height
            ax.fill_between(
                t,
                np.zeros_like(t),
                height,
                alpha=alpha,
                color=density_color,
                hatch=hatch,
                label=f"y={i}",
            )
    if plot_density_ticks:
        def _plot_ticks(fi, height=0.01, shift=0.0):
            # fi = get_ticks_from_density(dens, t, n_ticks = n_ticks)
            ax.vlines(fi, -height + shift, +height + shift, color='black', clip_on=False, zorder = 101)

        n_ticks = 100
        tickH = 0.01
        fs, ys = diagram['f_samp'][:n_ticks], diagram['y_samp'][:n_ticks]
        _plot_ticks(fs[ys == 0], tickH, -tickH)
        _plot_ticks(fs[ys == 1], tickH, +tickH)

    if "ce" in diagram.keys():
        ice = diagram["ce"]
        if "ce_ci_width" in diagram.keys():
            wid = diagram["ce_ci_width"]
            ax.text(0.05, 0.9, f"$\\mathrm{{smECE}}: {ice:.3f}\\pm {wid:.3f}$", zorder=1000)
        else:
            ax.text(0.05, 0.9, f"smECE: {ice:.3f}", zorder=1000)
    return fig, ax


def rel_diagram(
    f : npt.ArrayLike,
    y : npt.ArrayLike,
    **kwargs):
    """Compute and plot the reliability diagram for predictions f_i and labels y_i.
    This convenience function simply composes prepare_rel_diagram and plot_rel_diagram.
    """
    diagram = prepare_rel_diagram(f, y, **kwargs)
    return plot_rel_diagram(diagram, **kwargs)
