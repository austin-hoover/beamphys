"""Functions for point clouds."""
import numpy as np
import scipy.interpolate
import scipy.spatial
import scipy.stats

from psdist import ap
from psdist.utils import array_like
from psdist.utils import centers_from_edges
from psdist.utils import cov2corr
from psdist.utils import random_selection


# Analysis
# ------------------------------------------------------------------------------
def mean(X):
    """Compute mean (centroid).

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.

    Returns
    -------
    ndarray, shape (d,)
        The centroid coordinates.
    """
    return np.mean(X, axis=0)


def cov(X):
    """Compute covariance matrix.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.

    Returns
    -------
    ndarray, shape (d, d)
        The covariance matrix of second-order moments.
    """
    return np.cov(X.T)


def corr(X):
    """Compute correlation matrix.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.

    Returns
    -------
    ndarray, shape (d, d)
        The correlation matrix.
    """
    return cov2corr(np.cov(X.T))


def get_radii(X):
    return np.linalg.norm(X, axis=1)


def get_ellipsoid_radii(X):
    Sigma_inv = np.linalg.inv(np.cov(X.T))
    func = lambda point: np.sqrt(np.linalg.multi_dot([point.T, Sigma_inv, point]))
    return transform(X, func)


def enclosing_sphere(X, axis=None, fraction=1.0):
    """Scales sphere until it contains some fraction of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The distribution is projected onto this axis before proceeding. The
        ellipsoid is defined in this subspace.
    fraction : float
        Fraction of points in sphere.

    Returns
    -------
    radius : float
        The sphere radius.
    """
    radii = np.sort(get_radii(project(X, axis)))
    index = int(np.round(X.shape[0] * fraction)) - 1
    return radii[index]


def enclosing_ellipsoid(X, axis=None, fraction=1.0):
    """Scale the rms ellipsoid until it contains some fraction of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The distribution is projected onto this axis before proceeding. The
        ellipsoid is defined in this subspace.
    fraction : float
        Fraction of points enclosed.

    Returns
    -------
    float
        The ellipsoid "radius" (x^T Sigma^-1 x) relative to the rms ellipsoid.
    """
    radii = np.sort(get_ellipsoid_radii(project(X, axis)))
    index = int(np.round(X.shape[0] * fraction)) - 1
    return radii[index]



# Statistical distance
# ------------------------------------------------------------------------------

def maximum_mean_discrepancy(X, Y, sigma=1.0, method="spectral", L=1000, loop=True):
    """Estimate the maximum mean discrepancy (MMD) using a Gaussian kernel.
    
    Parameters
    ----------
    X : ndarray, shape (m, d)
    Y : ndarray, shape (n, d)
    sigma : float
    method : {"direct", "spectral"}
    L : int
    loop : bool
    """
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X.shape[1] != Y.shape[1]")
        
    # Stack the coordinate arrays.
    m, d = X.shape
    n, d = Y.shape
    X = np.vstack([X, Y])
    
    # Compute the coefficient vector.
    c = np.zeros(m + n)
    c[:m] = +1.0 / m
    c[m:] = -1.0 / n

    if method == "direct":
        c = c[:, None]
        kernel = np.exp(-(scipy.spatial.distance_matrix(X, X) ** 2) / (2.0 * sigma))
        return np.sqrt(np.sum(np.dot(c, c.T) * kernel))
    
    elif method == "spectral":
        omegas = np.random.normal(size=(L, X.shape[1]))
        result = 0.0
        for l in range(L):
            result += np.square(np.abs(np.real(np.sum(c * np.exp(1.0j * np.sum(omegas[l, :] * X[:, :], axis=1))))))
        return np.sqrt((1.0 / L) * result)  

    
    
# Transformation
# ------------------------------------------------------------------------------


def project(X, axis=None):
    """Axis-aligned projection. (Just calls `X[:, axis]`.)

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple[int], length l
        The axis on which to project the points.

    Returns
    -------
    ndarray, shape (n, l)
        The points projected onto the specified axis.
    """
    if axis is None:
        axis = tuple(np.arange(X.shape[1]))
    if array_like(axis) and len(axis) > X.shape[1]:
        raise ValueError("Invalid projection axis.")
    return X[:, axis]


def transform(X, func=None, **kws):
    """Apply a nonlinear transformation.

    This function just calls `np.apply_along_axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    function : callable
        Function applied to each point in X. Call signature is
        `function(point, **kws)` where `point` is an n-dimensional
        point given by one row of `X`.
    **kws
        Key word arguments for

    Returns
    -------
    ndarray, shape (n, d)
        The transformed distribution.
    """
    return np.apply_along_axis(lambda point: func(point, **kws), 1, X)


def transform_linear(X, M):
    """Apply a linear transformation.

    This function just calls `np.apply_along_axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    M : ndarray, shape (d, d)
        A linear transfer matrix.

    Returns
    -------
    ndarray, shape (n, d)
        The transformed distribution.
    """
    func = lambda point: np.matmul(M, point)
    return transform(X, lambda point: np.matmul(M, point))


def shift(X, delta=0.0):
    return X + delta


def scale(X, factor=1.0):
    return X * factor


def slice_planar(X, axis=None, center=None, width=None):
    """Return points within a planar slice.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        Slice axes. For example, (0, 1) will slice along the first and
        second axes of the array.
    center : ndarray, shape (d,)
        The center of the box.
    width : ndarray, shape (d,)
        The width of the box along each axis.

    Returns
    -------
    ndarray, shape (?, d)
        The points within the box.
    """
    k, n = X.shape
    if type(axis) is int:
        axis = (axis,)
    if type(center) in [int, float]:
        center = np.full(n, center)
    if type(width) in [int, float]:
        width = np.full(n, width)
    center = np.array(center)
    width = np.array(width)
    limits = list(zip(center - 0.5 * width, center + 0.5 * width))
    conditions = []
    for j, (umin, umax) in zip(axis, limits):
        conditions.append(X[:, j] > umin)
        conditions.append(X[:, j] < umax)
    idx = np.logical_and.reduce(conditions)
    return X[idx, :]


def slice_sphere(X, axis=None, rmin=0.0, rmax=None):
    """Return points within a spherical shell slice.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the sphere.
    rmin, rmax : float
        Inner/outer radius of spherical shell.

    Returns
    -------
    ndarray, shape (?, d)
        The points within the sphere.
    """
    if rmax is None:
        rmax = np.inf
    radii = get_radii(project(X, axis))
    idx = np.logical_and(radii > rmin, radii < rmax)
    return X[idx, :]


def slice_ellipsoid(X, axis=None, rmin=0.0, rmax=None):
    """Return points within an ellipsoidal shell slice.

    The ellipsoid is defined by the covariance matrix of the
    distribution.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the ellipsoid.
    rmin, rmax : list[float]
        Min/max "radius" (x^T Sigma^-1 x). relative to covariance matrix.

    Returns
    -------
    ndarray, shape (?, d)
        Points within the shell.
    """
    if rmax is None:
        rmax = np.inf
    radii = get_ellipsoid_radii(project(X, axis))
    idx = np.logical_and(rmin < radii, radii < rmax)
    return X[idx, :]


def slice_contour(X, axis=None, lmin=0.0, lmax=1.0, interp=True, **hist_kws):
    """Return points within a contour shell slice.

    The slice is defined by the density contours in the subspace defined by
    `axis`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    axis : tuple
        The subspace in which to define the density contours.
    lmin, lmax : list[float]
        If `f` is the density in the subspace defined by `axis`, then we select
        points where lmin <= f / max(f) <= lmax.
    interp : bool
        If True, compute the histogram, then interpolate and evaluate the
        resulting function at each point in `X`. Otherwise we keep track
        of the indices in which each point lands when it is binned,
        and accept the point if it's bin has a value within fmin and fmax.
        The latter is a lot slower.

    Returns
    -------
    ndarray, shape (?, d)
        Points within the shell.
    """
    _X = project(X, axis)
    hist, edges = histogram(_X, **hist_kws)
    hist = hist / np.max(hist)
    centers = [0.5 * (e[:-1] + e[1:]) for e in edges]
    if interp:
        fint = scipy.interpolate.RegularGridInterpolator(
            centers,
            hist,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        values = fint(_X)
        idx = np.logical_and(lmin <= values, values <= lmax)
    else:
        valid_indices = np.vstack(
            np.where(np.logical_and(lmin <= hist, hist <= lmax))
        ).T
        indices = np.vstack(
            [np.digitize(_X[:, k], edges[k]) for k in range(_X.shape[1])]
        ).T
        idx = []
        for i in range(len(indices)):
            if indices[i].tolist() in valid_indices.tolist():
                idx.append(i)
    return X[idx, :]


def norm_xxp_yyp_zzp(X, scale_emittance=False):
    """Normalize x-px, y-py, z-pz, ...

    Parameters
    ----------
    X : ndarray, shape (n, 2d)
        Coordinates of n points in 2d-dimensional phase space.
    scale_emittance : bool
        Whether to divide the coordinates by the square root of the rms emittance.

    Returns
    -------
    Xn : ndarray, shape (n, 2d)
        Normalized phase space coordinate array.
    """
    if X.shape[1] % 2 != 0:
        raise ValueError("X must have an even number of columns.")
    Sigma = np.cov(X.T)
    Xn = np.zeros(X.shape)
    for i in range(0, X.shape[1], 2):
        sigma = Sigma[i : i + 2, i : i + 2]
        alpha, beta = ap.twiss(sigma)
        Xn[:, i] = X[:, i] / np.sqrt(beta)
        Xn[:, i + 1] = (np.sqrt(beta) * X[:, i + 1]) + (alpha * X[:, i] / np.sqrt(beta))
        if scale_emittance:
            eps = ap.apparent_emittance(sigma)
            Xn[:, i : i + 2] = Xn[:, i : i + 2] / np.sqrt(eps)
    return Xn


def decorrelate(X):
    """Remove cross-plane correlations by permuting (x, x'), (y, y'), (z, z') pairs.

    Parameters
    ----------
    X : ndarray, shape (n, 2d)
        Coordinates of n points in 2d-dimensional space.

    Returns
    -------
    ndarray, shape (n, 2d)
        The decorrelated coordinates.
    """
    if X.shape[1] % 2 != 0:
        raise ValueError("X must have even number of columns.")
    for i in range(0, X.shape[1], 2):
        idx = np.random.permutation(np.arange(X.shape[0]))
        X[:, i : i + 2] = X[idx, i : i + 2]
    return X


def downsample(X, samples):
    """Select a random subset of points.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    samples : int or float
        The number of samples to keep If less than 1, specifies
        the fraction of points.

    Returns
    -------
    ndarray, shape (<= n, d)
        The downsampled coordinate array.
    """
    samples = min(samples, X.shape[0])
    idx = random_selection(np.arange(X.shape[0]), samples)
    return X[idx, :]


# Density estimation
# ------------------------------------------------------------------------------


def histogram_bin_edges(X, bins=10, limits=None):
    """Multi-dimensional histogram bin edges."""
    if not array_like(bins):
        bins = X.shape[1] * [bins]
    if not array_like(limits):
        limits = X.shape[1] * [limits]
    return [
        np.histogram_bin_edges(X[:, i], bins[i], limits[i]) 
        for i in range(X.shape[1])
    ]


def histogram(X, bins=10, limits=None, centers=False):
    """Multi-dimensional histogram."""
    edges = histogram_bin_edges(X, bins=bins, limits=limits)
    hist, edges = np.histogramdd(X, edges)
    if centers:
        return hist, [centers_from_edges(e) for e in edges]
    else:
        return hist, edges


def gaussian_kde(X, **kws):
    """Gaussian kernel density estimation (KDE).

    This function just calls `scipy.stats.gaussian_kde`.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Coordinates of n points in d-dimensional space.
    **kws
        Key word arguments

    Returns
    -------
    estimator : scipy.stats.gaussian_kde
        The density estimator.
    """
    return scipy.stats.gaussian_kde(X.T, **kws)
