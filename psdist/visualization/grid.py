import numpy as np
import proplot as pplt

import psdist.image
import psdist.cloud
import psdist.utils
import psdist.visualization.cloud as vis_cloud
import psdist.visualization.image as vis_image
from psdist.visualization.visualization import plot1d


class JointGrid:
    """Grid for joint plots.

    https://seaborn.pydata.org/generated/seaborn.JointGrid.html

    Attributes
    -----------
    fig : proplot.figure.Figure
        The main figure.
    ax : proplot.gridspec.SubplotGrid
        The main axis.
    ax_panel_x, ax_panel_y : proplot.gridspec.SubplotGrid
        The panel (marginal) axes on the top and right.
    """

    def __init__(
        self, panel_kws=None, panel_fmt_kws_x=None, panel_fmt_kws_y=None, **fig_kws
    ):
        """Constructor.

        marg_kws : dict
            Key word arguments for `ax.panel`.
        marg_fmt_kws_x, marg_fmt_kws_y : dict
            Key word arguments for `ax.format` for each of the panel axs.
        **fig_kws
            Key word arguments passed to `proplot.subplots`.
        """
        self.fig, self.ax = pplt.subplots(**fig_kws)

        if panel_kws is None:
            panel_kws = dict()
        panel_kws.setdefault("space", 0.0)
        self.ax_panel_x = self.ax.panel("t", **panel_kws)
        self.ax_panel_y = self.ax.panel("r", **panel_kws)
        self.panel_axs = [self.ax_panel_x, self.ax_panel_y]
        if panel_fmt_kws_x is None:
            panel_fmt_kws_x = dict()
        if panel_fmt_kws_y is None:
            panel_fmt_kws_y = dict()
        panel_fmt_kws = [panel_fmt_kws_x, panel_fmt_kws_y]
        for i, key in enumerate(["ylim", "xlim"]):
            panel_fmt_kws[i].setdefault(key, (0.0, 1.1))
        for ax, kws in zip(self.panel_axs, panel_fmt_kws):
            kws.setdefault("xspineloc", "neither")
            kws.setdefault("yspineloc", "neither")
            ax.format(**kws)

    def plot_cloud(self, X, marg_hist_kws=None, marg_kws=None, **kws):
        """Plot a 2D point cloud.

        Parameters
        ----------
        X : ndarray, shape (k, 2)
            Coordinates of k points in 2-dimensional space.
        marg_hist_kws : dict
            Key word arguments passed to `np.histogram` for 1D histograms.
        marg_kws : dict
            Key word arguments passed to `visualization.plot1d`.
        **kws
            Key word arguments passed to `visualization.image.plot2d.`
        """
        if marg_kws is None:
            marg_kws = dict()
        marg_kws.setdefault("kind", "step")
        marg_kws.setdefault("lw", 1.0)
        if marg_hist_kws is None:
            marg_hist_kws = dict()
        marg_hist_kws.setdefault("bins", "auto")
        kws.setdefault("kind", "hist")
        if kws["kind"] == "hist":
            kws.setdefault("mask", True)
            kws.setdefault("bins", marg_hist_kws["bins"])
        if kws["kind"] != "scatter":
            kws.setdefault("colorbar_kw", dict())
            kws["colorbar_kw"].setdefault("pad", 2.0)
        for axis in range(2):
            profile, edges = np.histogram(X[:, axis], **marg_hist_kws)
            profile = profile / np.max(profile)
            centers = psdist.utils.centers_from_edges(edges)
            plot1d(
                centers, profile, ax=self.panel_axs[axis], flipxy=bool(axis), **marg_kws
            )
        vis_cloud.plot2d(X, ax=self.ax, **kws)

    def plot_image(self, f, coords=None, marg_kws=None, **kws):
        """Plot a 2D image.

        Parameters
        ----------
        f : ndarray
            An n-dimensional image.
        coords : list[ndarray]
            Coordinates along each dimension of `f`.
        marg_kws : dict
            Key word arguments passed to `visualization.plot1d`.
        **kws
            Key word arguments passed to `visualization.image.plot2d.`
        """
        kws.setdefault("colorbar_kw", dict())
        kws["colorbar_kw"].setdefault("pad", 2.0)
        if marg_kws is None:
            marg_kws = dict()
        marg_kws.setdefault("color", "black")
        marg_kws.setdefault("kind", "step")
        marg_kws.setdefault("lw", 1.0)
        if coords is None:
            coords = [np.arange(f.shape[axis]) for axis in range(f.ndim)]
        _out = vis_image.plot2d(f, coords=coords, ax=self.ax, **kws)
        fx = psdist.image.project(f, axis=0)
        fy = psdist.image.project(f, axis=0)
        fx = fx / np.max(fx)
        fy = fy / np.max(fy)
        for axis in range(2):
            profile = psdist.image.project(f, axis)
            profile = profile / np.max(profile)
            plot1d(
                coords[axis],
                profile,
                ax=self.panel_axs[axis],
                flipxy=bool(axis),
                **marg_kws,
            )
        return _out

    def colorbar(self, mappable, **kws):
        """Add a colorbar."""
        kws.setdefault("loc", "r")
        kws.setdefault("pad", 2.0)
        self.fig.colorbar(mappable, **kws)


class CornerGrid:
    """Grid for corner plots.

    * https://seaborn.pydata.org/generated/seaborn.PairGrid.html
    * https://corner.readthedocs.io/en/latest/
    * https://pandas.pydata.org/docs/reference/api/pandas.plotting.scatter_matrix.html

    Attributes
    ----------
    fig : proplot.figure.Figure
        The main figure.
    axs : proplot.gridspec.SubplotGrid
        The subplot axes.
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
        n=4, 
        diag=True, 
        diag_height_frac=0.65, 
        limits=None, 
        labels=None, 
        corner=True,
        xspineloc="bottom", 
        yspineloc="left",
        diag_yspineloc="neither",
        **fig_kws
    ):
        """
        Parameters
        ----------
        n : int
            The number of rows/columns.
        diag : bool
            Whether to include diagonal subplots (univariate plots). If False,
            we have an (n - 1) x (n - 1) grid instead of an n x n grid.
        diag_height_frac : float
            This reduces the height of the diagonal plots relative to the ax
            height.
        diag_norm : {"max"}
            Normalize the diagonal plots so that maximum is 1.
        limits : list[tuple], length n
            The (min, max) for each dimension. (These can be set later.)
        labels : list[str]
            The label for each dimension. (These can be set later.)
        corner : bool
            Whether to hide the upper-triangular subplots.
        xspineloc, yspineloc : str
            Spine locations on all subplots {"left", "right", "top", "bottom", "neither", "both"}.
        diag_yspineloc : str
            Y spine location of diagonal subplots.
        **fig_kws
            Key word arguments passed to `pplt.subplots()`.
        """
        # Create figure.
        self.corner = corner
        self.new = True
        self.n = n
        self.diag = diag
        self.diag_height_frac = diag_height_frac
        self.start = int(self.diag)
        self.nrows = self.ncols = self.n
        if not self.diag:
            self.nrows = self.nrows - 1
            self.ncols = self.ncols - 1
        self.fig_kws = fig_kws
        self.fig_kws.setdefault("figwidth", 1.5 * self.nrows)
        self.fig_kws.setdefault("aligny", True)
        self.fig, self.axs = pplt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            sharex=1,
            sharey=1,
            spanx=False,
            spany=False,
            **self.fig_kws,
        )
        # Collect diagonal/off-diagonal subplots and indices.
        self.diag_axs = []
        self.offdiag_axs = []
        self.offdiag_axs_u = []
        self.diag_indices = []
        self.offdiag_indices = []
        self.offdiag_indices_u = []
        if self.diag:
            for i in range(self.n):
                self.diag_axs.append(self.axs[i, i])
                self.diag_indices.append(i)
            for i in range(1, self.n):
                for j in range(i):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_axs_u.append(self.axs[j, i])
                    self.offdiag_indices.append((j, i))
                    self.offdiag_indices_u.append((i, j))
        else:
            for i in range(self.n - 1):
                for j in range(i + 1):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_axs_u.append(self.axs[j, i])
                    self.offdiag_indices.append((j, i + 1))
                    self.offdiag_indices_u.append((i + 1, j))

        # Set limits and labels.
        self.limits = limits
        if limits is not None:
            self.set_limits(limits)
        self.labels = labels
        if labels is not None:
            self.set_labels(labels)

        # Formatting
        if self.corner:
            for i in range(self.nrows):
                for j in range(self.ncols):
                    if j > i:
                        self.axs[i, j].axis("off")
        for i in range(self.nrows):
            self.axs[:-1, i].format(xticklabels=[])
            self.axs[i, 1:].format(yticklabels=[])
        for ax in self.axs:
            ax.format(xspineloc=xspineloc, yspineloc=yspineloc)
        for ax in self.diag_axs:
            ax.format(yspineloc=diag_yspineloc)
        self.axs.format(
            xtickminor=True, ytickminor=True, xlocator=("maxn", 3), ylocator=("maxn", 3)
        )
        self.set_limits(limits)
        for ax in self.diag_axs:
            ax.format(ylim=(0.0, 1.01))

    def get_labels(self):
        """Return the n plot labels."""
        if self.diag:
            labels = [ax.get_xlabel() for ax in self.diag_axs]
        else:
            labels = [self.axs[-1, i].get_xlabel() for i in range(self.n - 1)]
            labels = labels + [self.axs[-1, 0].get_ylabel()]
        return labels

    def set_labels(self, labels):
        """Set the n plot labels."""
        for ax, label in zip(self.axs[-1, :], labels):
            ax.format(xlabel=label)
        for ax, label in zip(self.axs[self.start :, 0], labels[1:]):
            ax.format(ylabel=label)
        self.labels = labels

    def get_limits(self):
        """Return the n plot limits. (min, max)"""
        if self.diag:
            limits = [ax.get_xlim() for ax in self.diag_axs]
        else:
            limits = [self.axs[-1, i].get_xlim() for i in range(self.n - 1)]
            limits = limits + [self.axs[-1, 0].get_ylim()]
        return limits

    def set_limits(self, limits=None, expand=False):
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
            for i in range(self.axs.shape[1]):
                self.axs[:, i].format(xlim=limits[i])
            for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
                ax.format(ylim=limits[axis[1]])
            for ax, axis in zip(self.offdiag_axs_u, self.offdiag_indices_u):
                ax.format(ylim=limits[axis[1]])
        self.limits = self.get_limits()
        
    def plot_diag(self, x, y, axis=0, **kws):
        y = (y / np.max(y)) * self.diag_height_frac
        plot1d(x, y, ax=self.diag_axs[axis], **kws)

    def plot_image(
        self,
        f,
        coords=None,
        prof_edge_only=False,
        update_limits=True,
        diag_kws=None,
        **kws,
    ):
        """Plot an n-dimensional image.

        Parameters
        ----------
        f : ndarray
            An n-dimensional image.
        coords : list[ndarray]
            Coordinates along each axis of the grid (if `data` is an image).
        prof_edge_only : bool
            If plotting profiles on top of images (on off-diagonal subplots), whether
            to plot x profiles only in bottom row and y profiles only in left column.
        update_limits : bool
            Whether to extend the existing plot limits.
        diag_kws : dict
            Key word argument passed to `visualization.plot1d`.
        **kws
            Key word arguments pass to `visualization.image.plot2d`
        """
        if diag_kws is None:
            diag_kws = dict()
        diag_kws.setdefault("color", "black")
        diag_kws.setdefault("lw", 1.0)
        diag_kws.setdefault("kind", "step")
        diag_kws.setdefault("scale", None)
        
        kws.setdefault("kind", "pcolor")
        kws.setdefault("profx", False)
        kws.setdefault("profy", False)

        if coords is None:
            coords = [np.arange(f.shape[i]) for i in range(f.ndim)]

        if update_limits:
            edges = [psdist.utils.edges_from_centers(c) for c in coords]
            limits = [(np.min(e), np.max(e)) for e in edges]
            self.set_limits(limits, expand=(not self.new))
        self.new = False

        # Univariate plots.
        for ax, axis in zip(self.diag_axs, self.diag_indices):
            profile = psdist.image.project(f, axis=axis)
            self.plot_diag(coords[axis], profile, axis=axis, **diag_kws)

        # Bivariate plots.
        profx, profy = [kws.pop(key) for key in ("profx", "profy")]
        for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
            if prof_edge_only:
                if profx:
                    kws["profx"] = axis[1] == self.n - 1
                if profy:
                    kws["profy"] = axis[0] == 0
            _f = psdist.image.project(f, axis=axis)
            _f = _f / np.max(_f)
            _coords = [coords[i] for i in axis]
            vis_image.plot2d(_f, coords=_coords, ax=ax, **kws)
            
    def plot_cloud(
        self,
        X,
        limits=None,
        bins="auto",
        autolim_kws=None,
        prof_edge_only=False,
        lower=True,
        upper=True,
        update_limits=True,
        diag_kws=None,
        **kws,
    ):
        """Plot an n-dimensional point cloud.

        Parameters
        ----------
        X : ndarray, shape (k, n)
            Coordinates of k points in n-dimensional space.
        limits : list[tuple], length n
            The (min, max) axis limits.
        bins : 'auto', int, list[int]
            The number of bins along each dimension (if plot type requires histogram
            computation). If int or 'auto', applies to all dimensions. Currently
            the histogram is computed with limits based on the data min/max, not
            the plot limits.
        prof_edge_only : bool
            If plotting profiles on top of images (on off-diagonal subplots), whether
            to plot x profiles only in bottom row and y profiles only in left column.
        lower, upper, bool
            Whether to plot on the lower or upper triangular subplots (or both).
        update_limits : bool
            Whether to extend the existing plot limits.
        diag_kws : dict
            Key word argument passed to `visualization.plot1d`.
        **kws
            Key word arguments pass to `visualization.cloud.plot2d`
        """
        n = X.shape[1]
        if diag_kws is None:
            diag_kws = dict()
        diag_kws.setdefault("color", "black")
        diag_kws.setdefault("lw", 1.0)
        diag_kws.setdefault("kind", "step")
        kws.setdefault("kind", "hist")
        kws.setdefault("profx", False)
        kws.setdefault("profy", False)

        if limits is None:
            if autolim_kws is None:
                autolim_kws = dict()
            autolim_kws.setdefault("pad", 0.1)
            limits = vis_cloud.auto_limits(X, **autolim_kws)
        if update_limits:
            self.set_limits(limits, expand=(not self.new))
        limits = self.get_limits()
        self.new = False

        if not psdist.utils.array_like(bins):
            bins = n * [bins]

        # Univariate plots. Remember histogram bins and use them for 2D histograms.
        edges = []
        for ax, axis in zip(self.diag_axs, self.diag_indices):
            if psdist.utils.array_like(bins[axis]):
                _edges = bins[axis]
            else:
                _edges = np.histogram_bin_edges(X[:, axis], bins[axis], limits[axis])
            edges.append(_edges)
            heights, _ = np.histogram(X[:, axis], _edges)
            centers = psdist.utils.centers_from_edges(_edges)
            self.plot_diag(centers, heights, axis=axis, **diag_kws)

        # Bivariate plots:
        profx, profy = [kws.pop(key) for key in ("profx", "profy")]
        if lower:
            for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
                if prof_edge_only:
                    if profx:
                        kws["profx"] = axis[1] == self.n - 1
                    if profy:
                        kws["profy"] = axis[0] == 0
                if kws["kind"] in ["hist", "contour", "contourf"]:
                    kws["bins"] = [edges[axis[0]], edges[axis[1]]]
                vis_cloud.plot2d(X[:, axis], ax=ax, **kws)
        if upper and not self.corner:
            for ax, axis in zip(self.offdiag_axs_u, self.offdiag_indices_u):
                if kws["kind"] in ["hist", "contour", "contourf"]:
                    kws["bins"] = [edges[axis[0]], edges[axis[1]]]
                vis_cloud.plot2d(X[:, axis], ax=ax, **kws)


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

    Attributes
    ----------
    fig : proplot.figure.Figure
        The main figure.
    axs : proplot.gridspec.SubplotGrid
        The subplot axes.
    _axs : proplot.figure.Figure
        The subplot axes on the main panel.
    _axs_marg_x, _axs_marg_y, _axs_marg_xy : proplot.gridspec.SubplotGrid
        The subplot axes on the marginal panels.
    """

    def __init__(
        self,
        nrows=9,
        ncols=9,
        space=0.0,
        gap=2.0,
        marginals=True,
        annotate=True,
        annotate_kws_view=None,
        annotate_kws_slice=None,
        slice_label_height=0.22,
        **fig_kws,
    ):
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
            Key word arguments for `pplt.subplots`.
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
            self.annotate_kws_view = dict()
        self.annotate_kws_view.setdefault("color", "black")
        self.annotate_kws_view.setdefault("xycoords", "axes fraction")
        self.annotate_kws_view.setdefault("horizontalalignment", "center")
        self.annotate_kws_view.setdefault("verticalalignment", "center")

        self.annotate_kws_slice = annotate_kws_slice
        if self.annotate_kws_slice is None:
            self.annotate_kws_slice = dict()
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

        self.fig, self.axs = pplt.subplots(**fig_kws)

        self._axs = self.axs[:-1, :-1]
        self._axs_marg_x = []
        self._axs_marg_y = []
        if self.marginals:
            self._axs_marg_x = self.axs[-1, :]
            self._axs_marg_y = self.axs[:, -1]
        self._ax_marg_xy = self.axs[-1, -1]

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
        anchors = (self.axs[i, self.ncols // 2], self.axs[self.nrows // 2, i])
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

    def get_ind_slice(self):
        """Return slice indices from latest plot call."""
        return self.ind_slice

    def get_axis_slice(self):
        """Return slice axis from latest plot call."""
        return self.axis_slice

    def get_axis_view(self):
        """Return view axis from latest plot call."""
        return self.axis_view

    def set_limits(self, limits):
        """Set the plot limits."""
        for ax in self.axs:
            ax.format(xlim=limits[0], ylim=limits[1])

    def plot_image(
        self,
        f,
        coords=None,
        labels=None,
        axis_view=(0, 1),
        axis_slice=(2, 3),
        pad=0.0,
        debug=False,
        **kws,
    ):
        """Plot an n-dimensional image.

        Parameters
        ----------
        f : ndarray
            An n-dimensional image.
        coords : list[ndarray]
            Coordinates along each axis of the grid (if `data` is an image).
        labels : list[str], length n
            Label for each dimension.
        axis_view, axis_slice : 2-tuple of int
            The axis to view (plot) and to slice.
        pad : int, float, list
            This determines the start/stop indices along the sliced dimensions. If
            0, space the indices along axis `i` uniformly between 0 and `f.shape[i]`.
            Otherwise, add a padding equal to `int(pad[i] * f.shape[i])`. So, if
            the shape=10 and pad=0.1, we would start from 1 and end at 9.
        debug : bool
            Whether to print debugging messages.
        **kws
            Key word arguments pass to `visualization.image.plot2d`
        """
        # Setup
        # -----------------------------------------------------------------------
        if f.ndim < 4:
            raise ValueError(f"f.ndim = {f.ndim} < 4")
        if coords is None:
            coords = [np.arange(s) for s in f.shape]
        self.axis_view = axis_view
        self.axis_slice = axis_slice

        # Compute 4D/3D/2D projections.
        _f = psdist.image.project(f, axis_view + axis_slice)
        _fx = psdist.image.project(f, axis_view + axis_slice[:1])
        _fy = psdist.image.project(f, axis_view + axis_slice[1:])
        _fxy = psdist.image.project(f, axis_view)

        # Compute new coords and labels.
        _coords = [coords[i] for i in axis_view + axis_slice]
        _labels = None
        if labels is not None:
            _labels = [labels[i] for i in axis_view + axis_slice]

        # Select slice indices.
        if type(pad) in [float, int]:
            pad = len(axis_slice) * [pad]
        ind_slice = []
        for i, steps, _pad in zip(axis_slice, [self.nrows, self.ncols], pad):
            start = int(_pad * f.shape[i])
            stop = f.shape[i] - 1 - start
            if (stop - start) < steps:
                raise ValueError(f"f.shape[{i}] < number of slice indices requested.")
            ind_slice.append(np.linspace(start, stop, steps).astype(int))
        ind_slice = tuple(ind_slice)
        self.ind_slice = ind_slice

        if debug:
            print("Slice indices:")
            for ind in ind_slice:
                print(ind)

        # Slice the 4D projection. The axes order was already handled by `project`;
        # the first two axes are the view axes and the last two axes are the
        # slice axes.
        axis_view = (0, 1)
        axis_slice = (2, 3)
        idx = 4 * [slice(None)]
        for axis, ind in zip(axis_slice, ind_slice):
            idx[axis] = ind
            _f = _f[tuple(idx)]
            idx[axis] = slice(None)

        # Slice the 3D projections.
        _fx = _fx[:, :, ind_slice[0]]
        _fy = _fy[:, :, ind_slice[1]]

        # Slice coords.
        for i, ind in zip(axis_slice, ind_slice):
            _coords[i] = _coords[i][ind]

        # Normalize each distribution.
        _f = psdist.image.process(_f, norm="max")
        _fx = psdist.image.process(_fx, norm="max")
        _fy = psdist.image.process(_fy, norm="max")
        _fxy = psdist.image.process(_fxy, norm="max")

        if debug:
            print("_f.shape =", _f.shape)
            print("_fx.shape =", _fx.shape)
            print("_fy.shape =", _fy.shape)
            print("_fxy.shape =", _fxy.shape)
            for i in range(_f.ndim):
                assert _f.shape[i] == len(_coords[i])

        # Add dimension labels to the figure.
        if self.annotate and _labels is not None:
            self._annotate(
                labels=_labels,
                slice_label_height=self.slice_label_height,
                annotate_kws_view=self.annotate_kws_view,
                annotate_kws_slice=self.annotate_kws_slice,
            )

        # Plotting
        # -----------------------------------------------------------------------
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[self.nrows - 1 - i, j]
                idx = psdist.image.slice_idx(
                    _f.ndim, axis=axis_slice, ind=[(j, j + 1), (i, i + 1)]
                )
                vis_image.plot2d(
                    psdist.image.project(_f[idx], axis_view),
                    coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
        if self.marginals:
            for i, ax in enumerate(reversed(self.axs[:-1, -1])):
                vis_image.plot2d(
                    _fy[:, :, i],
                    coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
            for i, ax in enumerate(self.axs[-1, :-1]):
                vis_image.plot2d(
                    _fx[:, :, i],
                    [_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
            vis_image.plot2d(
                _fxy,
                coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                ax=self.axs[-1, -1],
                **kws,
            )

    def _plot_cloud(
        self,
        X,
        labels=None,
        bins_slice="auto",
        axis_view=(0, 1),
        axis_slice=(2, 3),
        pad=0.0,
        debug=False,
        autolim_kws=None,
        **kws,
    ):
        """Plot an n-dimensional point cloud.

        NOTE: this is not currently working... for the time being, it is recommended
        to generate a 4D histogram and then call `plot_image`.

        Parameters
        ----------
        X : ndarray, shape (k, n)
            Coordinates of k points in n-dimensional space.
        labels : list[str], length n
            Label for each dimension.
        bins_slice : int, list[int], list[ndarray]
            Specifies the bins used for slicing in `axis_slice`. The bin range
            is determined by the min/max point in `X`.
        axis_view, axis_slice : 2-tuple of int
            The axis to view (plot) and to slice.
        pad : int, float, list
            Fractional padding added to the start/stop indices for the bins in the sliced
            dimensions.
        debug : bool
            Whether to print debugging messages.
        **kws
            Key word arguments pass to `visualization.cloud.plot2d`
        """

        warnings.warn(
            "This is not currently working. For the time being, it is recommended to generate a 4D histogram and call `plot_image`."
        )

        # Setup
        # -----------------------------------------------------------------------
        if X.shape[1] < 4:
            raise ValueError(f"X.shape[1] = {X.shape[1]} < 4")
        self.axis_view = axis_view
        self.axis_slice = axis_slice

        edges_slice = psdist.cloud.histogram_bin_edges(X[:, axis_slice], bins_slice)

        # Select slice indices.
        if type(pad) in [float, int]:
            pad = len(axis_slice) * [pad]
        ind_slice = []
        for i, (steps, _pad) in enumerate(zip([self.nrows, self.ncols], pad)):
            start = int(_pad * len(edges_slice[i]))
            stop = len(edges_slice[i]) - 1 - start
            if (stop - start) < steps:
                raise ValueError(f"Too many slices requested.")
            ind_slice.append(np.linspace(start, stop, steps + 1).astype(int))
        ind_slice = tuple(ind_slice)
        self.ind_slice = ind_slice

        # Slice the bin indices.
        edges_slice = [e[ind] for e, ind in zip(edges_slice, ind_slice)]

        # Add dimension labels to the figure.
        if labels is not None:
            if self.annotate and labels is not None:
                self._annotate(
                    labels=[labels[k] for k in axis_view + axis_slice],
                    slice_label_height=self.slice_label_height,
                    annotate_kws_view=self.annotate_kws_view,
                    annotate_kws_slice=self.annotate_kws_slice,
                )

        # Plotting
        # -----------------------------------------------------------------------
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[self.nrows - 1 - i, j]
                _X = psdist.cloud.slice_planar(
                    X,
                    axis=axis_slice,
                    center=[
                        0.5 * (edges_slice[0][j] + edges_slice[0][j + 1]),
                        0.5 * (edges_slice[1][i] + edges_slice[1][i + 1]),
                    ],
                    width=[
                        np.abs(edges_slice[0][j] - edges_slice[1][j + 1]),
                        np.abs(edges_slice[1][i] - edges_slice[1][i + 1]),
                    ],
                )
                vis_cloud.plot2d(_X[:, axis_slice], ax=ax, **kws)
