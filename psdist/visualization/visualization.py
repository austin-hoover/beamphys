from ipywidgets import interact
from ipywidgets import interactive
from ipywidgets import widgets
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np
import proplot as pplt
import scipy.optimize

import psdist.image
import psdist.discrete
import psdist.utils
import psdist.visualization.discrete as psv_discrete
import psdist.visualization.image as psv_image


def ellipse(c1=1.0, c2=1.0, angle=0.0, center=(0, 0), ax=None, **kws):
    """Plot ellipse with semi-axes `c1`,`c2` tilted `angle`radians below the x axis."""
    kws.setdefault("fill", False)
    kws.setdefault("color", "black")
    width = 2.0 * c1
    height = 2.0 * c2
    return ax.add_patch(
        patches.Ellipse(center, width, height, -np.degrees(angle), **kws)
    )


def circle(r=1.0, center=(0.0, 0.0), ax=None, **kws):
    """Plot a circle."""
    return ellipse(r, r, center=center, ax=ax, **kws)


def rms_ellipse_dims(Sigma, axis=(0, 1)):
    """Return dimensions of projected rms ellipse.

    Parameters
    ----------
    Sigma : ndarray, shape (2n, 2n)
        The phase space covariance matrix.
    axis : 2-tuple
        The axis on which to project the covariance ellipsoid. Example: if the
        axes are {x, xp, y, yp}, and axis=(0, 2), then the four-dimensional
        ellipsoid is projected onto the x-y plane.
    ax : plt.Axes
        The ax on which to plot.

    Returns
    -------
    c1, c2 : float
        The ellipse semi-axis widths.
    angle : float
        The tilt angle below the x axis [radians].
    """
    i, j = axis
    sii, sjj, sij = Sigma[i, i], Sigma[j, j], Sigma[i, j]
    angle = -0.5 * np.arctan2(2 * sij, sii - sjj)
    sin, cos = np.sin(angle), np.cos(angle)
    sin2, cos2 = sin**2, cos**2
    c1 = np.sqrt(abs(sii * cos2 + sjj * sin2 - 2 * sij * sin * cos))
    c2 = np.sqrt(abs(sii * sin2 + sjj * cos2 + 2 * sij * sin * cos))
    return c1, c2, angle


def rms_ellipse(Sigma=None, center=None, level=1.0, ax=None, **ellipse_kws):
    """Plot RMS ellipse from 2 x 2 covariance matrix."""
    if type(level) not in [list, tuple, np.ndarray]:
        level = [level]
    c1, c2, angle = rms_ellipse_dims(Sigma)
    for level in level:
        _c1 = c1 * level
        _c2 = c2 * level
        ellipse(_c1, _c2, angle=angle, center=center, ax=ax, **ellipse_kws)
    return ax


def linear_fit(x, y):
    """Return (yfit, slope, intercept) from linear fit."""

    def fit(x, slope, intercept):
        return slope * x + intercept

    popt, pcov = scipy.optimize.curve_fit(fit, x, y)
    slope, intercept = popt
    yfit = fit(x, *popt)
    return yfit, slope, intercept


def plot1d(x, y, ax=None, offset=0.0, flipxy=False, kind="line", **kws):
    """Convenience function for one-dimensional line/step/bar plots."""
    func = ax.plot
    if kind in ["line", "step"]:
        if flipxy:
            func = ax.plotx
        else:
            func = ax.plot
        if kind == "step":
            kws.setdefault("drawstyle", "steps-mid")
    elif kind in ["linefilled", "stepfilled"]:
        if flipxy:
            func = ax.fill_betweenx
        else:
            func = ax.fill_between
        kws.setdefault("alpha", 1.0)
        if kind == "stepfilled":
            kws.setdefault("step", "mid")
    elif kind == "bar":
        if flipxy:
            func = ax.barh
        else:
            func = ax.bar

    # Handle offset
    if kind == "bar":
        kws["left" if flipxy else "bottom"] = offset * np.ones(len(x))
        return func(x, y, **kws)
    elif kind in ["linefilled", "stepfilled"]:
        return func(x, offset, y + offset, **kws)
    return func(x, y + offset, **kws)


class CornerGrid:
    """Grid for corner plots.
    
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
    limits : list[tuple], length n
        The (min, max) for each dimension. (These can be set later.)
    labels : list[str]
        The label for each dimension. (These can be set later.)
    **fig_kws
        Key word arguments passed to `pplt.subplots()`.
    """
    def __init__(
        self, 
        n=4, 
        diag=True, 
        diag_height_frac=0.65, 
        limits=None, 
        labels=None,
        **fig_kws
    ):
        # Create figure.
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
        self.limits = limits
        if limits is not None:
            self.set_limits(limit)
        self.labels = labels
        if labels is not None:
            self.set_labels(labels)

        # Collect diagonal/off-diagonal subplots and indices.
        self.diag_axs = []
        self.offdiag_axs = []
        self.diag_indices = []
        self.offdiag_indices = []
        if self.diag:
            for i in range(self.n):
                self.diag_axs.append(self.axs[i, i])
                self.diag_indices.append(i)
            for i in range(1, self.n):
                for j in range(i):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_indices.append((j, i))
        else:
            for i in range(self.n - 1):
                for j in range(i + 1):
                    self.offdiag_axs.append(self.axs[i, j])
                    self.offdiag_indices.append((j, i + 1))

        # Formatting
        for i in range(self.nrows):
            for j in range(self.ncols):
                if j > i:
                    self.axs[i, j].axis('off')
        for i in range(self.nrows):
            self.axs[:-1, i].format(xticklabels=[])
            self.axs[i, 1:].format(yticklabels=[])
        for ax in self.axs:
            ax.format(xspineloc='bottom', yspineloc='left')
        for ax in self.diag_axs:
            ax.format(yspineloc='neither')
        self.axs.format(
            xtickminor=True, ytickminor=True, 
            xlocator=('maxn', 3), ylocator=('maxn', 3)
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
            for i, lim in enumerate(limits[1:], start=self.start):
                self.axs[i, : (i + 1 - self.start)].format(ylim=lim)
        self.limits = self.get_limits()

    def plot_image(
        self,
        f,
        coords=None,
        prof_edge_only=False,
        update_limits=True,
        diag_kws=None,
        **kws
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
        diag_kws.setdefault('color', 'black')
        diag_kws.setdefault('lw', 1.0)
        diag_kws.setdefault('kind', 'step')
        kws.setdefault('kind', 'pcolor')
        kws.setdefault('profx', False)
        kws.setdefault('profy', False)

        if coords is None:
            coords = [np.arange(f.shape[i]) for i in range(f.ndim)]

        if update_limits:
            edges = [psdist.utils.edges_from_centers(c) for c in coords]
            limits = [(np.min(e), np.max(e)) for e in edges]
            self.set_limits(limits, expand=update_limits)

        # Univariate plots.
        for ax, axis in zip(self.diag_axs, self.diag_indices):
            profile = psdist.image.project(f, axis=axis)
            profile = profile / np.max(profile)
            profile = profile * self.diag_height_frac
            plot1d(coords[axis], profile, ax=ax, **diag_kws)

        # Bivariate plots.
        profx, profy = [kws.pop(key) for key in ('profx', 'profy')]
        for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
            if prof_edge_only:
                if profx: 
                    kws['profx'] = axis[1] == self.n - 1
                if profy:
                    kws['profy'] = axis[0] == 0
            _f = psdist.image.project(f, axis=axis)
            _f = _f / np.max(_f)
            _coords = [coords[i] for i in axis]
            psv_image.plot2d(_f, coords=_coords, ax=ax, **kws)

    def plot_discrete(
        self,
        X,
        limits=None,
        bins='auto',
        autolim_kws=None,
        prof_edge_only=False,
        update_limits=True,
        diag_kws=None,
        **kws
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
        update_limits : bool
            Whether to extend the existing plot limits.
        diag_kws : dict
            Key word argument passed to `visualization.plot1d`.
        **kws
            Key word arguments pass to `visualization.discrete.plot2d`
        """
        n = X.shape[1]
        if diag_kws is None:
            diag_kws = dict()
        diag_kws.setdefault('color', 'black')
        diag_kws.setdefault('lw', 1.0)
        diag_kws.setdefault('kind', 'step')
        kws.setdefault('kind', 'hist')
        kws.setdefault('profx', False)
        kws.setdefault('profy', False)
        
        if np.ndim(bins) == 0:
            bins = n * [bins]

        if limits is None:
            if autolim_kws is None:
                autolim_kws = dict()
            limits = psv_discrete.auto_limits(X, **autolim_kws)
        if update_limits:
            self.set_limits(limits, expand=True)

        # Univariate plots. Remember histogram bins and use them for 2D histograms.
        for ax, axis in zip(self.diag_axs, self.diag_indices):
            heights, edges = np.histogram(X[:, axis], bins[axis], limits[axis])
            heights = heights / np.max(heights)
            heights = heights * self.diag_height_frac
            centers = psdist.utils.centers_from_edges(edges)
            plot1d(centers, heights, ax=ax, **diag_kws)

        # Bivariate plots:
        profx, profy = [kws.pop(key) for key in ('profx', 'profy')]
        for ax, axis in zip(self.offdiag_axs, self.offdiag_indices):
            if prof_edge_only:
                if profx: 
                    kws['profx'] = axis[1] == self.n - 1
                if profy:
                    kws['profy'] = axis[0] == 0
            if kws['kind'] in ['hist', 'contour', 'contourf']:
                kws['bins'] = bins
            psv_discrete.plot2d(X[:, axis], ax=ax, **kws)

            
class JointGrid:
    """Grid for joint plots."""
    def __init__(self):
        return NotImplementedError
    
    
class SliceGrid:
    """Grid for slice matrix plots."""
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
        label_height=0.22,
        **fig_kws
    ):
        """Matrix of bivariate plots as two other dimensions are sliced.

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

        Parameters
        ----------
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
        label_height : float
            Tweaks the position of the slice dimension labels.
        **fig_kws
            Key word arguments for `pplt.subplots`.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.space = space
        self.gap = gap
        self.marginals = marginals
        self.annotate = annotate
        self.label_height = label_height
        self.fig_kws = fig_kws
        self.axis_slice = None
        self.axis_view = None
        self.slice_indices = None
        
        self.annotate_kws_view = annotate_kws_view
        if self.annotate_kws_view is None:
            self.annotate_kws_view = dict()
        self.annotate_kws_view.setdefault('color', 'black')
        self.annotate_kws_view.setdefault('xycoords', 'axes fraction')
        self.annotate_kws_view.setdefault('horizontalalignment', 'center')
        self.annotate_kws_view.setdefault('verticalalignment', 'center')
        
        self.annotate_kws_slice = annotate_kws_slice
        if self.annotate_kws_slice is None:
            self.annotate_kws_slice = dict()
        self.annotate_kws_slice.setdefault('color', 'black')
        self.annotate_kws_slice.setdefault('xycoords', 'axes fraction')
        self.annotate_kws_slice.setdefault('horizontalalignment', 'center')
        self.annotate_kws_slice.setdefault('verticalalignment', 'center')
        self.annotate_kws_slice.setdefault('arrowprops', dict(arrowstyle='->', color='black'))

        fig_kws.setdefault('figwidth', 8.5 * (ncols / 13.0))
        fig_kws.setdefault('share', False)
        fig_kws['ncols'] = ncols + 1 if marginals else ncols
        fig_kws['nrows'] = nrows + 1 if marginals else nrows
        hspace = nrows * [space]
        wspace = ncols * [space]
        if marginals:
            hspace[-1] = wspace[-1] = gap
        else:
            hspace = hspace[:-1]
            wspace = wspace[:-1]
        fig_kws['hspace'] = hspace
        fig_kws['wspace'] = wspace
        
        self.fig, self.axs = pplt.subplots(**fig_kws)
            
        self._axs = self.axs[:-1, :-1]
        self._axs_marg_x = []
        self._axs_marg_y = []
        if self.marginals:
            self._axs_marg_x = self.axs[-1, :]
            self._axs_marg_y = self.axs[:, -1]

    def _annotate(
        self, labels=None,
        label_height=0.22,
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
        
            
        # ilast = -2 if self.marginals else -1  # index of last ax in main panel
        i = -1  - int(self.marginals)
        anchors = (self.axs[i, self.ncols // 2], self.axs[self.nrows // 2, i])
        anchors[0].annotate(labels[2], xy=(0.5, -label_height), **annotate_kws_slice)
        anchors[1].annotate(labels[3], xy=(1.0 + label_height, 0.5), **annotate_kws_slice)
        for arrow_direction in (1.0, -1.0):
            anchors[0].annotate(
                '',
                xy=(0.5 + arrow_direction * arrow_length, -label_height),
                xytext=(0.5 + arrow_direction * text_length, -label_height),
                **annotate_kws_slice,
            )
            anchors[1].annotate(
                '',
                xy=(1.0 + label_height, 0.5 + arrow_direction * arrow_length),
                xytext=(1.0 + label_height, 0.5 + arrow_direction * text_length),
                **annotate_kws_slice,
            )
            
    def get_slice_indices(self):
        """Return slice indices from latest plot call."""
        return self.slice_indices
    
    def get_axis_slice(self):
        """Return slice axis from latest plot call."""
        return self.axis_slice
    
    def get_axis_view(self):
        """Return view axis from latest plot call."""
        return self.axis_view
                        
    def plot_image(
        self,
        f,
        coords=None,
        labels=None,
        axis_view=(0, 1),
        axis_slice=(2, 3),
        pad=0.0,
        debug=False,
        **kws
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
            raise ValueError(f'f.ndim = {f.ndim} < 4')
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
            ind_slice.append(np.linspace(start, stop, steps + 1).astype(int))
        ind_slice = tuple(ind_slice)
        self.slice_indices = ind_slice

        if debug:
            print('Slice indices:')
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
        _f = psdist.image.process(_f, norm='max')
        _fx = psdist.image.process(_fx, norm='max')
        _fy = psdist.image.process(_fy, norm='max')
        _fxy = psdist.image.process(_fxy, norm='max')

        if debug:
            print('_f.shape =', _f.shape)
            print('_fx.shape =', _fx.shape)
            print('_fy.shape =', _fy.shape)
            print('_fxy.shape =', _fxy.shape)
            for i in range(_f.ndim):
                assert _f.shape[i] == len(_coords[i])
                
        # Add dimension labels to the figure.
        if self.annotate and _labels is not None:
            self._annotate(
                labels=_labels,
                label_height=self.label_height, 
                annotate_kws_view=self.annotate_kws_view,
                annotate_kws_slice=self.annotate_kws_slice,
            )

        # Plotting
        # -----------------------------------------------------------------------
        for i in range(self.nrows):
            for j in range(self.ncols):
                ax = self.axs[self.nrows - 1 - i, j]
                idx = psdist.image.make_slice(_f.ndim, axis=axis_slice, ind=[(j, j + 1), (i, i + 1)])
                psv_image.plot2d(
                    psdist.image.project(_f[idx], axis_view),
                    coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
        if self.marginals:
            for i, ax in enumerate(reversed(self.axs[:-1, -1])):
                psv_image.plot2d(
                    _fy[:, :, i],
                    coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
            for i, ax in enumerate(self.axs[-1, :-1]):
                psv_image.plot2d(
                    _fx[:, :, i],
                    [_coords[axis_view[0]], _coords[axis_view[1]]],
                    ax=ax,
                    **kws,
                )
            psv_image.plot2d(
                _fxy,
                coords=[_coords[axis_view[0]], _coords[axis_view[1]]],
                ax=self.axs[-1, -1],
                **kws,
            )