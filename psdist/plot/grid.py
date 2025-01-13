import warnings
from typing import Callable
from typing import Union

import numpy as np
import matplotlib.pyplot as plt

from ..hist import Histogram
from ..hist import Histogram1D
from ..core import histogram as _histogram
from ..core import limits as _get_limits
from ..utils import array_like
from .core import plot
from .core import plot_1d
from .hist import plot as plot_hist
from .hist import plot_1d as plot_hist_1d


class JointGrid:
    """Grid for joint plots.

    Attributes
    -----------
    fig : proplot.figure.Figure
        The main figure.
    ax : proplot.gridspec.SubplotGrid
        The main axis.
    ax_panel_x, ax_panel_y : proplot.gridspec.SubplotGrid
        The marginal (panel) axes on the top and right.
    """

    def __init__(
        self,
        limits: np.ndarray = None,
        panel_kws: dict = None,
        panel_fmt_kws: dict = None,
        panel_fmt_kws_x: dict = None,
        panel_fmt_kws_y: dict = None,
        **fig_kws,
    ) -> None:
        """Constructor.

        panel_kws : dict
            Key word arguments for `ax.panel`.
        panel_fmt_kws, panel_fmt_kws_x, panel_fmt_kws_y : dict
            Key word arguments for `ax.format` for each of the marginal (panel) axs.
        **fig_kws
            Key word arguments passed to `proplot.subplots`.
        """
        self.fig, self.ax = uplt.subplots(**fig_kws)

        self.limits = limits
        if self.limits is not None:
            self.set_limits(self.limits)

        if panel_kws is None:
            panel_kws = {}
        if panel_fmt_kws is None:
            panel_fmt_kws = {}
        if panel_fmt_kws_x is None:
            panel_fmt_kws_x = {}
        if panel_fmt_kws_y is None:
            panel_fmt_kws_y = {}

        panel_fmt_kws_x.setdefault("xspineloc", "bottom")
        panel_fmt_kws_x.setdefault("yspineloc", "left")
        panel_fmt_kws_y.setdefault("xspineloc", "bottom")
        panel_fmt_kws_y.setdefault("yspineloc", "left")

        self.ax_panel_x = self.ax.panel("t", **panel_kws)
        self.ax_panel_y = self.ax.panel("r", **panel_kws)
        self.panel_axs = [self.ax_panel_x, self.ax_panel_y]
        for ax in self.panel_axs:
            ax.format(**panel_fmt_kws)
        self.panel_axs[0].format(**panel_fmt_kws_x)
        self.panel_axs[1].format(**panel_fmt_kws_y)

        self.default_panel_plot_kws = {
            "color": "black",
            "kind": "step",
            "lw": 1.5,
            "scale": "density",
        }

        self.default_panel_hist_kws = {"bins": 64}

        self.frozen_limits = False

    def set_limits(self, limits: np.ndarray) -> None:
        self.limits = limits
        self.axs.format(xlim=limits[0], ylim=limits[1])

    def reset_limits(self) -> None:
        self.set_limits(self.limits)

    def freeze_limits(self) -> None:
        self.frozen_limits = True

    def unfreeze_limits(self) -> None:
        self.frozen_limits = False

    def wrapup(self) -> None:
        if self.frozen_limits:
            self.reset_limits()

    def plot(self, points: np.ndarray, **kws):
        return self.plot_points(points, **kws)

    def plot_points(
        self,
        points: np.ndarray,
        panel_hist_kws: dict = None,
        panel_plot_kws: dict = None,
        **kws,
    ) -> None:
        """Plot 2D points.

        Parameters
        ----------
        points: np.ndarray, shape (..., 2)
            Particle coordinates
        panel_hist_kws : dict
            Key word arguments passed to `psdist.points.histogram`.
        panel_plot_kws : dict
            Key word arguments passed to `plot_hist_1d`.
        **kws
            Key word arguments passed to `plot_points.`
        """
        if panel_plot_kws is None:
            panel_plot_kws = {}

        if panel_hist_kws is None:
            panel_hist_kws = {}

        for key, val in self.default_panel_hist_kws.items():
            panel_hist_kws.setdefault(key, val)

        for key, val in self.default_panel_plot_kws.items():
            panel_plot_kws.setdefault(key, val)

        # Plot marginal distributions
        for axis in range(2):
            hist_proj = _histogram(points[:, axis], **panel_hist_kws)
            bins = hist_proj.size

            _plot_hist_1d(
                hist_proj,
                ax=self.panel_axs[axis],
                orientation=("horizontal" if bool(axis) else "vertical"),
                **panel_plot_kws,
            )

        # Plot joint distribution
        kws.setdefault("kind", "hist")
        if kws["kind"] != "scatter":
            if "bins" in kws:
                panel_hist_kws.setdefault("bins", kws["bins"])
            kws.setdefault("bins", bins)
            kws.setdefault("colorbar_kw", {})
            kws["colorbar_kw"].setdefault("pad", 2.0)

        _plot_points(points, ax=self.ax, **kws)

        self.wrapup()

    def plot_hist(
        self,
        hist: Histogram,
        panel_plot_kws: dict = None,
        **kws,
    ) -> None:
        """Plot a two-dimensional histogram.

        Parameters
        ----------
        hist: Histogram
            A two-dimensional histogram.
        panel_plot_kws : dict
            Key word arguments passed to `plot_hist_1d`.
        **kws
            Key word arguments passed to `plot_hist.`
        """
        if panel_plot_kws is None:
            panel_plot_kws = {}

        for key, val in self.default_panel_plot_kws.items():
            panel_plot_kws.setdefault(key, val)

        kws.setdefault("kind", "pcolor")
        kws.setdefault("colorbar_kw", {})
        kws["colorbar_kw"].setdefault("pad", 2.0)

        # Plot marginal distributions
        for axis in range(2):
            hist_proj = hist.project(axis)
            _plot_hist_1d(
                hist_proj,
                ax=self.panel_axs[axis],
                orientation=("horizontal" if bool(axis) else "vertical"),
                **panel_plot_kws,
            )

        # Plot joint distribution
        _plot_hist(hist, ax=self.ax, **kws)

        self.wrapup()

    def colorbar(self, mappable, **kws):
        """Add a colorbar."""
        kws.setdefault("loc", "r")
        kws.setdefault("pad", 2.0)
        self.fig.colorbar(mappable, **kws)


class CornerGrid:
    """Grid for corner plots.

    Attributes
    ----------
    fig, axs : plt.Figure, plt.Axes
        Figure and axes.
    diag_axs : list[proplot.gridspec.SubplotGrid]
        The axes for diagonal (univariate) subplots. Can be empty.
    offdiag_axs : list[proplot.gridspec.SubplotGrid]
        The axes for off-diagonal (bivariate) subplots.
    diag_indices : list[int]
        The index of the dimension plotted on each diagonal subplot.
    offdiag_indices : list[2-tuple of int]
        Indices of the dimensions plotted on each off-diagonal subplot.
    """

    def __init__(
        self,
        ndim: int,
        diag: bool = True,
        diag_shrink: float = 1.0,
        diag_rspine: bool = False,
        diag_lspine: bool = False,
        diag_share: bool = False,
        diag_frozen: bool = False,
        limits: list[tuple[float, float]] = None,
        labels: list[str] = None,
        corner: bool = True,
        **fig_kws,
    ) -> None:
        """
        Parameters
        ----------
        ndim : int
            The number of rows/columns.
        diag : bool
            Whether to include diagonal subplots (univariate plots). If False,
            we have an (N - 1) x (N - 1) grid instead of an N x N grid.
        diag_shrink : float in range [0, 1]
            Scales the maximum y value of the diagonal profile plots.
        diag_rspine, diag_lspine : bool
            Whether to include right/left spine on diagonal subplots (if `corner`).
        diag_share : bool
            Whether to share diagonal axis limits; i.e., whether we can compare
            the areas under the diagonal profile plots.
        diag_frozen : bool
            If true, don't adjust the y limits on diagonal axes after the first
            distribution is plotted. Otherwise, always adjust to keep the
            histograms in view.
        limits : list[tuple], length n
            The (min, max) for each dimension. (These can be set later.)
        labels : list[str]
            The label for each dimension. (These can be set later.)
        corner : bool
            Whether to hide the upper-triangular subplots.
        **fig_kws
            Key word arguments passed to `plt.subplots()`.
        """        
        # Create figure.
        self.new = True
        self.ndim = self.nrows = self.ncols = ndim
        self.corner = corner

        self.diag = diag
        self.diag_shrink = diag_shrink
        self.diag_rspine = diag_rspine
        self.diag_lspine = diag_lspine
        self.diag_share = diag_share
        self.diag_ymin = None
        self.diag_yscale = self.ndim * [None]
        self.diag_frozen = diag_frozen

        if not self.diag:
            self.nrows = self.nrows - 1
            self.ncols = self.ncols - 1

        self.fig_kws = fig_kws
        self.fig_kws.setdefault("figsize", (1.25 * self.nrows, 1.25 * self.ncols))
        self.fig_kws.setdefault("constrained_layout", True)

        self.fig, self.axs = plt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            sharex=False,
            sharey=False,
            **self.fig_kws,
        )
        # self.fig.align_ylabels(self.axs[:, 0])

        # Collect diagonal/off-diagonal subplots and indices.
        self.diag_axs = []
        self.offdiag_axs = []
        self.offdiag_axs_u = []

        self.diag_indices = []
        self.offdiag_indices = []
        self.offdiag_indices_u = []

        if self.diag:
            for i in range(self.ndim):
                self.diag_axs.append(self.axs[i, i])
                self.diag_indices.append(i)
            for i in range(1, self.ndim):
                for j in range(i):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_axs_u.append(self.axs[j, i])
                    self.offdiag_indices.append((j, i))
                    self.offdiag_indices_u.append((i, j))
        else:
            for i in range(self.ndim - 1):
                for j in range(i + 1):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_indices.append((j, i + 1))

        # Set limits and labels.
        self.limits = limits
        if limits is not None:
            self.set_limits(limits)

        self.labels = labels
        if labels is not None:
            self.set_labels(labels)

        # Formatting
        if self.corner or not self.diag:
            for i in range(self.nrows):
                for j in range(self.ncols):
                    if j > i:
                        self.axs[i, j].axis("off")

        for i in range(self.ndim - 1):
            for j in range(self.ndim):
                self.axs[i, j].set_xticklabels([])
            
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[i, j]
                if i != self.nrows - 1:
                    ax.set_xticklabels([])
                if j != 0:
                    if not (i == j and self.diag_rspine and self.corner and self.diag):
                        ax.set_yticklabels([])

        for ax in self.axs.flat:
            for loc in ["top", "right"]:
                ax.spines[loc].set_visible(False)

        if self.corner:
            for ax in self.diag_axs:
                if not self.diag_lspine:
                    ax.spines["left"].set_visible(False)
                    # ax.set_yticks([])
                if not self.diag_rspine:
                    ax.spines["right"].set_visible(False)
                    # ax.set_yticks([])
                    
        # for ax in self.axs.flat:
        #     ax.set_xtickminor(True)
        #     ax.set_ytickminor(True)
        # self.axs.format(
        #     xtickminor=True, ytickminor=True, xlocator=("maxn", 3), ylocator=("maxn", 3)
        # )
        # self.set_diag_scale("linear")

    # def format_diag(self, **kws) -> None:
    #     """Format diagonal subplots."""
    #     for ax in self.diag_axs:
    #         ax.format(**kws)
    #     if not self.corner:
    #         for ax in self.diag_axs[1:]:
    #             ax.format(yticklabels=[])
    #     self._fake_diag_yticks()

    # def format_offdiag(self, **kws) -> None:
    #     """Format off-diagonal subplots."""
    #     for ax in [self.offdiag_axs + self.offdiag_axs_u]:
    #         ax.format(**kws)

    def get_labels(self) -> list[str]:
        """Return the dimension labels."""
        if self.diag:
            labels = [ax.get_xlabel() for ax in self.diag_axs]
        else:
            labels = [self.axs[-1, i].get_xlabel() for i in range(self.ndim - 1)]
            labels = labels + [self.axs[-1, 0].get_ylabel()]
        return labels

    def set_labels(self, labels: list[str]) -> None:
        """Set the dimension labels."""
        for ax, label in zip(self.axs[-1, :], labels):
            ax.format(xlabel=label)
        for ax, label in zip(self.axs[int(self.diag) :, 0], labels[1:]):
            ax.format(ylabel=label)
        if self.diag and not self.corner:
            self.axs[0, 0].format(ylabel=labels[0])
        self.labels = labels

    def get_limits(self) -> list[tuple[float, float]]:
        """Return the plot limits."""
        if self.diag:
            limits = [ax.get_xlim() for ax in self.diag_axs]
        else:
            limits = [self.axs[-1, i].get_xlim() for i in range(self.ndim - 1)]
            limits = limits + [self.axs[-1, 0].get_ylim()]
        return limits

    def set_limits(
        self, limits: list[tuple[float, float]], expand: bool = False
    ) -> None:
        """Set the plot limits.

        Parameters
        ----------
        limits : list[tuple], length n
            The (min, max) for each dimension.
        expand : bool
            If True, compare the proposed limits to the existing limits, expanding
            if the new limits are wider.
        """
        if limits is not None:
            if expand:
                limits = np.array(limits)
                limits_old = np.array(self.get_limits())
                mins = np.minimum(limits[:, 0], limits_old[:, 0])
                maxs = np.maximum(limits[:, 1], limits_old[:, 1])
                limits = list(zip(mins, maxs))

            if self.diag:
                for i in range(self.ndim):
                    for j in range(self.ndim):
                        if i != j:
                            if (j < i) or (not self.corner):
                                self.axs[i, j].set_ylim(limits[i])
            else:
                for i in range(self.ndim - 1):
                    for ax in self.axs[i, :]:
                        ax.set_ylim(limits[i + 1])

            for i in range(self.ncols):
                for ax in self.axs[:, i]:
                    ax.set_xlim(limits[i])

        self.limits = self.get_limits()

    def get_default_diag_kws(self, diag_kws: dict = None) -> dict:
        if diag_kws is None:
            diag_kws = {}
        diag_kws.setdefault("color", "black")
        diag_kws.setdefault("lw", 1.0)
        diag_kws.setdefault("kind", "step")
        diag_kws.setdefault("scale", "density")
        return diag_kws

    def _fake_diag_yticks(self) -> None:
        """The yticks on the (0, 0) subplot correspond to the other subplots in the row.

        Source: pandas.plotting.scatterplot_matrix.
        """
        if self.corner or not self.diag:
            return
        limits = self.limits
        if limits is None:
            limits = self.get_limits()

        lim1 = limits[0]
        locs = self.axs[0, 0].xaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = self.axs[0, 0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        self.axs[0, 0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            locs = locs.astype(int)
        self.axs[0, 0].yaxis.set_ticklabels(locs)

    def _force_non_negative_diag_ymin(self) -> None:
        """Force diagonal ymins to be at least zero."""
        if any([ax.get_ylim()[0] < 0.0 for ax in self.diag_axs]):
            for ax in self.diag_axs:
                ymin, ymax = ax.get_ylim()
                ax.set_ylim(0.0, ymax)

    # def set_diag_scale(self, scale: str = "linear", pad: float = 0.05) -> None:
    #     """Set diagonal axis scale.

    #     Parameters
    #     ----------
    #     scale : {"linear", "log"}
    #         If "linear", scale runs from 0 to 1. If "log", scale runs from half the
    #         minimum plotted value to 1.
    #     pad: float
    #         Padding applied to the y axis limit.
    #     """
    #     if scale == "linear":
    #         ymin = 0.0
    #         ymax = 1.0 / self.diag_shrink
    #         delta = ymax - ymin
    #         ymax = ymin + delta * (1.0 + pad)
    #         self.format_diag(yscale="linear", yformatter="auto", ymin=ymin, ymax=ymax)
    #     elif scale == "log":
    #         ymin = self.diag_ymin
    #         ymax = 1.0
    #         log_ymin = np.log10(ymin)
    #         log_ymax = np.log10(ymax)
    #         log_delta = log_ymax - log_ymin
    #         log_delta = log_delta / self.diag_shrink
    #         ymax = 10.0 ** (log_ymin + log_delta * (1.0 + pad))
    #         self.format_diag(yscale="log", yformatter="log", ymin=ymin, ymax=ymax)

    def plot_diag(self, hists: list[Histogram1D], **kws) -> None:
        """Compute one-dimensional histograms."""
        if "scale" in kws:
            kws.pop("scale")

        # Normalize histograms
        for hist in hists:
            hist.normalize()

        # Update diag_yscale
        if not (self.diag_frozen and not self.new):
            for axis in range(self.ndim):
                if self.diag_yscale[axis] is None:
                    self.diag_yscale[axis] = -np.inf
            if self.diag_share:
                max_value = np.max([np.max(hist.values) for hist in hists])
                self.diag_yscale = np.maximum(self.diag_yscale, max_value)
            else:
                max_values = np.array([np.max(hist.values) for hist in hists])
                self.diag_yscale = np.maximum(self.diag_yscale, max_values)

        # Plot histograms
        for axis, hist in enumerate(hists):

            yscale = self.diag_yscale[axis]
            if yscale is None:
                yscale = 1.0

            hist_scaled = hist.copy()
            hist_scaled.values /= yscale

            plot_hist_1d(hist_scaled, ax=self.diag_axs[axis], **kws)

        # Compute minimum positive value (for log scaling)
        for axis, hist in enumerate(hists):
            if not self.diag_ymin:
                self.diag_ymin = np.inf
            self.diag_ymin = min(self.diag_ymin, np.min(hist.values[hist.values > 0.0]))

    def plot_hist(
        self,
        hist: Histogram,
        prof_edge_only: bool = False,
        lower: bool = True,
        upper: bool = True,
        diag: bool = True,
        update_limits: bool = True,
        diag_kws: dict = None,
        **kws,
    ) -> None:
        """Plot an N-dimensional histogram.

        Parameters
        ----------
        hist: np.ndarray
            An N-dimensional histogram.
        prof_edge_only : bool
            If plotting profiles on top of images (on off-diagonal subplots), whether
            to plot x profiles only in bottom row and y profiles only in left column.
        lower, upper, diag : bool
            Whether to plot on the lower triangular, upper triangular, and/or diagonal subplots.
        update_limits : bool
            Whether to extend the existing plot limits.
        diag_kws : dict
            Key word argument passed to `plot_hist_1d`.
        **kws
            Key word arguments pass to `plot_hist`
        """
        kws.setdefault("kind", "pcolor")

        diag_kws = self.get_default_diag_kws(diag_kws)

        if update_limits:
            self.set_limits(hist.limits, expand=(not self.new))

        # Univariate plots
        if self.diag and diag:
            hists = [hist.project(axis) for axis in range(hist.ndim)]
            self.plot_diag(hists, **diag_kws)

        # Bivariate plots
        profx = kws.get("profx", False)
        profy = kws.pop("profy", False)

        if lower:
            for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
                hist_proj = hist.project(axis)

                if prof_edge_only:
                    if profx:
                        kws["profx"] = axis[1] == self.ndim - 1
                    if profy:
                        kws["profy"] = axis[0] == 0

                plot_hist(hist_proj, ax=ax, **kws)

        if upper and not self.corner:
            for ax, axis in zip(self.offdiag_axs_u, self.offdiag_indices_u):
                hist_proj = hist.project(axis)
                plot_hist(hist_proj, ax=ax, **kws)

        self._post_plot()

    def plot(self, points: np.ndarray, **kws) -> None:
        return self.plot_points(points, **kws)

    def plot_points(
        self,
        points: np.ndarray,
        limits: list[tuple[float, float]] = None,
        bins: int = 50,
        prof_edge_only: bool = False,
        lower: bool = True,
        upper: bool = True,
        diag: bool = True,
        autolim_kws: dict = None,
        update_limits: bool = True,
        diag_kws: dict = None,
        **kws,
    ) -> None:
        """Plot points.

        Parameters
        ----------
        points : np.ndarray, shape (..., N)
            Particle coordinates.
        limits : list[tuple[float, float]], length N
            The (min, max) axis limits.
        bins : 'auto', int, list[int]
            The number of bins along each dimension (if plot type requires histogram
            computation). If int or 'auto', applies to all dimensions.
        prof_edge_only : bool
            If plotting profiles on top of images (on off-diagonal subplots), whether
            to plot x profiles only in bottom row and y profiles only in left column.
        lower, upper, diag : bool
            Whether to plot on the lower triangular, upper triangular, and/or diagonal subplots.
        update_limits : bool
            Whether to extend the existing plot limits.
        diag_kws : dict
            Key word argument passed to `plot.plot_profile`.
        **kws
            Key word arguments pass to `plot.points.plot`
        """
        kws.setdefault("kind", "hist")

        diag_kws = self.get_default_diag_kws(diag_kws)

        if autolim_kws is None:
            autolim_kws = {}
        autolim_kws.setdefault("pad", 0.25)

        if limits is None:
            limits = _get_limits(points, **autolim_kws)

        if update_limits:
            self.set_limits(limits, expand=(not self.new))

        limits = self.get_limits()

        if not array_like(bins):
            bins = points.shape[1] * [bins]

        # Compute histogram bin edges.
        hists = []
        for axis in range(self.ndim):
            if array_like(bins[axis]):
                edges = bins[axis]
            else:
                edges = np.histogram_bin_edges(points[:, axis], bins[axis], limits[axis])
                
            hist = Histogram1D(edges=edges)
            hist.bin(points[:, axis])
            hists.append(hist)

        # Univariate plots.
        if self.diag and diag:
            self.plot_diag(hists, **diag_kws)

        # Bivariate plots:
        profx = kws.pop("profx", False)
        profy = kws.pop("profy", False)

        if lower:
            for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
                if prof_edge_only:
                    if profx:
                        kws["profx"] = axis[1] == self.ndim - 1
                    if profy:
                        kws["profy"] = axis[0] == 0

                if kws["kind"] in ["hist", "contour", "contourf"]:
                    kws["bins"] = [
                        hists[axis[0]].edges,
                        hists[axis[1]].edges,
                    ]

                plot(points[:, axis], ax=ax, **kws)

        if upper and not self.corner:
            for ax, axis in zip(self.offdiag_axs_u, self.offdiag_indices_u):
                if kws["kind"] in ["hist", "contour", "contourf"]:
                    kws["bins"] = [
                        hists[axis[0]].edges,
                        hists[axis[1]].edges,
                    ]

                plot(points[:, axis], ax=ax, **kws)

        self._post_plot()

    def _post_plot(self) -> None:
        self.new = False
        self._fake_diag_yticks()
        self._force_non_negative_diag_ymin()


class SliceGrid:
    """Grid for slice matrix plots (https://arxiv.org/abs/2301.04178).

    This plot is used to visualize four dimensions of a distribution f(x1, x2, x3, x4).

    The main panel is an nrows x ncols grid that shows f(x1, x2 | x3, x4) -- the
    x1-x2 distribution for a planar slice in x3-x4. Each subplot corresponds to a
    different location in the x3-x4 plane.

    The following is only included if `marginals` is True:

        The bottom panel shows the marginal 3D distribution f(x1, x2 | x3).

        The right panel shows the marginal 3D distribution f(x1, x2 | x4).

        The bottom right subplot shows the full projection f(x1, x2).

        The lone subplot on the bottom right shows f(x1, x2)l, the full projection
        onto the x1-x2 plane.

    To do
    -----
    * Option to flow from top left to bottom right.

    Attributes
    ----------
    fig : proplot.figure.Figure
        The main figure.
    axs : proplot.gridspec.SubplotGrid
        The subplot axes.
    _axs : proplot.figure.Figure
        The subplot axes on the main panel.
    _axs_panel_x, _axs_panel_y, _axs_panel_xy : proplot.gridspec.SubplotGrid
        The subplot axes on the marginal panels.
    """

    def __init__(
        self,
        nrows: int = 9,
        ncols: int = 9,
        space: float = 0.0,
        gap: float = 2.0,
        marginals: bool = True,
        annotate: bool = True,
        annotate_kws_view: dict = None,
        annotate_kws_slice: dict = None,
        slice_label_height: float = 0.22,
        **fig_kws,
    ) -> None:
        """Constructor.

        nrows, ncols : int
            The number of rows/colums in the figure.
        space : float
            Spacing between subplots.
        gap : float
            Gap between main and marginal panels.
        marginals : bool
            Whether to include the marginal panels. If they are not included, we just
            have an nrows x ncols grid.
        annotate : bool
            Whether to add dimension labels/arrows to the figure.
        annotate_kws_view, annotate_kws_slice : dict
            Key word arguments for figure text. The 'view' key words are for the view
            dimension labels; they are printed on top of one of the subplots. The
            'slice' key words are for the slice dimension labels; they are printed
            on the sides of the figure, between the main and marginal panels.
        slice_label_height : float
            Tweaks the position of slice labels. Need a better way to handle this.
        **fig_kws
            Key word arguments for `uplt.subplots`.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.space = space
        self.gap = gap
        self.marginals = marginals
        self.annotate = annotate
        self.slice_label_height = slice_label_height
        self.fig_kws = fig_kws
        self.axis_slice = None
        self.axis_view = None
        self.ind_slice = None

        self.annotate_kws_view = annotate_kws_view
        if self.annotate_kws_view is None:
            self.annotate_kws_view = {}
        self.annotate_kws_view.setdefault("color", "black")
        self.annotate_kws_view.setdefault("xycoords", "axes fraction")
        self.annotate_kws_view.setdefault("horizontalalignment", "center")
        self.annotate_kws_view.setdefault("verticalalignment", "center")

        self.annotate_kws_slice = annotate_kws_slice
        if self.annotate_kws_slice is None:
            self.annotate_kws_slice = {}
        self.annotate_kws_slice.setdefault("color", "black")
        self.annotate_kws_slice.setdefault("xycoords", "axes fraction")
        self.annotate_kws_slice.setdefault("horizontalalignment", "center")
        self.annotate_kws_slice.setdefault("verticalalignment", "center")
        self.annotate_kws_slice.setdefault(
            "arrowprops", dict(arrowstyle="->", color="black")
        )

        fig_kws.setdefault("figwidth", 8.5 * (ncols / 13.0))
        fig_kws.setdefault("share", True)
        fig_kws["ncols"] = ncols + 1 if marginals else ncols
        fig_kws["nrows"] = nrows + 1 if marginals else nrows
        hspace = nrows * [space]
        wspace = ncols * [space]
        if marginals:
            hspace[-1] = wspace[-1] = gap
        else:
            hspace = hspace[:-1]
            wspace = wspace[:-1]
        fig_kws["hspace"] = hspace
        fig_kws["wspace"] = wspace

        self.fig, self.axs = uplt.subplots(**fig_kws)

        self._axs = self.axs[:-1, :-1]
        self._axs_panel_x = []
        self._axs_panel_y = []
        if self.marginals:
            self._axs_panel_x = self.axs[-1, :]
            self._axs_panel_y = self.axs[:, -1]
        self._ax_panel_xy = self.axs[-1, -1]

    def _annotate(
        self,
        labels=None,
        slice_label_height=0.22,
        annotate_kws_view=None,
        annotate_kws_slice=None,
    ):
        """Add dimension labels and arrows."""
        # Label the view dimensions.
        for i, xy in enumerate([(0.5, 0.13), (0.12, 0.5)]):
            self.axs[0, 0].annotate(labels[i], xy=xy, **self.annotate_kws_view)

        # Label the slice dimensions. Print dimension labels with arrows like this:
        # "<----- x ----->" on the bottom and right side of the main panel.
        arrow_length = 2.5  # arrow length
        text_length = 0.15  # controls space between dimension label and start of arrow
        i = -1 - int(self.marginals)
        anchors = (
            self.axs[i, self.ncols // 2],
            self.axs[self.nrows // 2, i],
        )
        anchors[0].annotate(
            labels[2], xy=(0.5, -slice_label_height), **annotate_kws_slice
        )
        anchors[1].annotate(
            labels[3], xy=(1.0 + slice_label_height, 0.5), **annotate_kws_slice
        )
        for arrow_direction in (1.0, -1.0):
            anchors[0].annotate(
                "",
                xy=(0.5 + arrow_direction * arrow_length, -slice_label_height),
                xytext=(0.5 + arrow_direction * text_length, -slice_label_height),
                **annotate_kws_slice,
            )
            anchors[1].annotate(
                "",
                xy=(1.0 + slice_label_height, 0.5 + arrow_direction * arrow_length),
                xytext=(1.0 + slice_label_height, 0.5 + arrow_direction * text_length),
                **annotate_kws_slice,
            )

    def get_ind_slice(self) -> int:
        """Return slice indices from latest plot call."""
        return self.ind_slice

    def get_axis_slice(self) -> int:
        """Return slice axis from latest plot call."""
        return self.axis_slice

    def get_axis_view(self) -> tuple[int, ...]:
        """Return view axis from latest plot call."""
        return self.axis_view

    def set_limits(self, limits) -> None:
        """Set the plot limits."""
        for ax in self.axs:
            ax.format(xlim=limits[0], ylim=limits[1])

    def plot_hist(
        self,
        hist: Histogram,
        axis_view: tuple[int, int] = (0, 1),
        axis_slice: tuple[int, int] = (2, 3),
        labels: list[str] = None,
        pad: float = 0.0,
        debug: bool = False,
        **kws,
    ) -> None:
        """Plot a four-dimensional histogram.

        The first two dimensions are plotted as the last two are sliced.

        Parameters
        ----------
        hist: Histogram
            A four-dimensional histogram.
        axis_view, axis_slice : tuple[int, int]
        labels : list[str]
            Label for each dimension.
        pad : int, float, list
            This determines the start/stop indices along the sliced dimensions. If
            0, space the indices along axis `i` uniformly between 0 and `values.shape[i]`.
            Otherwise, add a padding equal to `int(pad[i] * values.shape[i])`. So, if
            the shape=10 and pad=0.1, we would start from 1 and end at 9.
        debug : bool
            Whether to print debugging messages.
        **kws
            Key word arguments pass to `plot.image.plot`
        """

        # Setup
        # -----------------------------------------------------------------------

        if hist.ndim < 4:
            raise ValueError(f"hist.ndim = {hist.ndim} < 4")

        self.axis_view = axis_view
        self.axis_slice = axis_slice

        # Compute 4D/3D/2D projections.
        _hist = hist.project(axis=(axis_view + axis_slice))
        _hist_x = hist.project(axis=(axis_view + axis_slice[:1]))
        _hist_y = hist.project(axis=(axis_view + axis_slice[1:]))
        _hist_xy = hist.project(axis=axis_view)

        # Get slice indices
        pad_factors = pad
        if type(pad) in [float, int]:
            pad_factors = len(axis_slice) * [pad_factors]

        ind_slice = []
        for axis, nsteps, pad_factor in zip(
            axis_slice, [self.nrows, self.ncols], pad_factors
        ):
            lo = hist.shape[axis] * pad_factor
            lo = int(lo)
            hi = hist.shape[axis] - 1 - lo

            if (hi - lo) < nsteps:
                raise ValueError(
                    f"values.shape[{i}] < number of slice indices requested."
                )

            if (hi - lo) == (nsteps - 1):
                ind_slice.append([int(i) for i in np.arange(nsteps)])
            else:
                ind_slice.append([int(i) for i in np.linspace(lo, hi, nsteps)])

        ind_slice = tuple(ind_slice)
        self.ind_slice = ind_slice

        if debug:
            print("debug slice indices:")
            for ind in ind_slice:
                print(ind)

        # Slice the 4D histogram.
        axis_view, axis_slice = (0, 1), (2, 3)
        for axis, ind in zip(axis_slice, ind_slice):
            _hist = _hist.slice(axis=axis, ind=ind)

        # Slice the 3D projections.
        _hist_x = _hist_x.slice(axis=2, ind=ind_slice[0])
        _hist_y = _hist_y.slice(axis=2, ind=ind_slice[1])

        # Scale all to unit maximum
        _hist.scale_max()
        _hist_x.scale_max()
        _hist_y.scale_max()
        _hist_xy.scale_max()

        if debug:
            print("debug _hist.shape =", _hist_x.shape)
            print("debug _hist_x.shape =", _hist_x.shape)
            print("debug _hist_y.shape =", _hist_y.shape)
            print("debug _hist_xy.shape =", _hist_xy.shape)

        # Get labels
        if labels is not None:
            labels = [labels[axis] for axis in axis_view + axis_slice]

        # Add dimension labels to the figure.
        if self.annotate and labels is not None:
            self._annotate(
                labels=labels,
                slice_label_height=self.slice_label_height,
                annotate_kws_view=self.annotate_kws_view,
                annotate_kws_slice=self.annotate_kws_slice,
            )

        # Plotting
        # -----------------------------------------------------------------------

        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[self.nrows - 1 - i, j]
                _hist_slice = _hist.slice(axis=axis_slice, ind=[j, i])
                _plot_hist(_hist_slice, ax=ax, **kws)

        if self.marginals:
            for i, ax in enumerate(reversed(self.axs[:-1, -1])):
                _hist_y_slice = _hist_y.slice(axis=2, ind=i)
                _plot_hist(_hist_y_slice, ax=ax, **kws)

            for i, ax in enumerate(self.axs[-1, :-1]):
                _hist_x_slice = _hist_x.slice(axis=2, ind=i)
                _plot_hist(_hist_x_slice, ax=ax, **kws)

            _plot_hist(_hist_xy, ax=self.axs[-1, -1], **kws)
