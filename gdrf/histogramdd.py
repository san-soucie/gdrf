import scipy.sparse as sparse
import numpy as np
import operator
import functools
import contextlib
from numpy.core import overrides
import torch
import collections

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')

# range is a keyword argument to many functions, so save the builtin so they can
# use it.
_range = range

def _get_outer_edges(a, range):
    """
    Determine the outer bin edges to use, from either the data or the range
    argument
    """
    if range is not None:
        first_edge, last_edge = range
        if first_edge > last_edge:
            raise ValueError(
                'max must be larger than min in range parameter.')
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "supplied range of [{}, {}] is not finite".format(first_edge, last_edge))
    elif a.size == 0:
        # handle empty arrays. Can't determine range, so use 0-1.
        first_edge, last_edge = 0, 1
    else:
        first_edge, last_edge = a.min(), a.max()
        if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            raise ValueError(
                "autodetected range of [{}, {}] is not finite".format(first_edge, last_edge))

    # expand empty range to avoid divide by zero
    if first_edge == last_edge:
        first_edge = first_edge - 0.5
        last_edge = last_edge + 0.5

    return first_edge, last_edge


def histogramdd(sample, bins=10):
    """
    Compute the multidimensional histogram of some data.
    Parameters
    ----------
    sample : (N, D) array, or (D, N) array_like
        The data to be histogrammed.
        Note the unusual interpretation of sample when an array_like:
        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramdd((X, Y, Z))``.
        The first form should be preferred.
    bins : sequence or int, optional
        The bin specification:
        * A sequence of arrays describing the monotonically increasing bin
          edges along each dimension.
        * The number of bins for each dimension (nx, ny, ... =bins)
        * The number of bins for all dimensions (nx=ny=...=bins).
    range : sequence, optional
        A sequence of length D, each an optional (lower, upper) tuple giving
        the outer bin edges to be used if the edges are not given explicitly in
        `bins`.
        An entry of None in the sequence results in the minimum and maximum
        values being used for the corresponding dimension.
        The default, None, is equivalent to passing a tuple of D None values.
    density : bool, optional
        If False, the default, returns the number of samples in each bin.
        If True, returns the probability *density* function at the bin,
        ``bin_count / sample_count / bin_volume``.
    normed : bool, optional
        An alias for the density argument that behaves identically. To avoid
        confusion with the broken normed argument to `histogram`, `density`
        should be preferred.
    weights : (N,) array_like, optional
        An array of values `w_i` weighing each sample `(x_i, y_i, z_i, ...)`.
        Weights are normalized to 1 if normed is True. If normed is False,
        the values of the returned histogram are equal to the sum of the
        weights belonging to the samples falling into each bin.
    Returns
    -------
    H : ndarray
        The multidimensional histogram of sample x. See normed and weights
        for the different possible semantics.
    edges : list
        A list of D arrays describing the bin edges for each dimension.
    See Also
    --------
    histogram: 1-D histogram
    histogram2d: 2-D histogram
    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape

    nbin = np.empty(D, int)
    edges = D * [None]
    dedges = D * [None]

    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D * [bins]

    # normalize the range argument
    range = (None,) * D
    # Create edge arrays
    for i in _range(D):
        if np.ndim(bins[i]) == 0:
            if bins[i] < 1:
                raise ValueError(
                    '`bins[{}]` must be positive, when an integer'.format(i))
            smin, smax = _get_outer_edges(sample[:, i], range[i])
            try:
                n = operator.index(bins[i])

            except TypeError as e:
                raise TypeError(
                    "`bins[{}]` must be an integer, when a scalar".format(i)
                ) from e

            edges[i] = np.linspace(smin, smax, n + 1)
        elif np.ndim(bins[i]) == 1:
            edges[i] = np.asarray(bins[i])
            if np.any(edges[i][:-1] > edges[i][1:]):
                raise ValueError(
                    '`bins[{}]` must be monotonically increasing, when an array'
                        .format(i))
        else:
            raise ValueError(
                '`bins[{}]` must be a scalar or 1d array'.format(i))

        nbin[i] = len(edges[i]) + 1  # includes an outlier on each end
        dedges[i] = np.diff(edges[i])

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in _range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in _range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    good_rows = np.logical_and.reduce([(x < nbin[i] - 1) & (x > 0) for i, x in enumerate(Ncount)])

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    Ncount_goodrows = [x[good_rows] - 1 for x in Ncount]

    hist = torch.sparse_coo_tensor(Ncount_goodrows, [1.0 for _ in Ncount_goodrows[0]], torch.Size(nbin - 2))
    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.

    return hist, edges

def main():
    x = np.random.randn(6, 5)
    histogramdd(x)

if __name__ == "__main__":
    main()
