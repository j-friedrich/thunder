from __future__ import division
from numpy import percentile
from thunder.extraction.block.base import BlockMethod, BlockAlgorithm
from thunder.extraction.source import Source


class BlockGroupLasso(BlockMethod):

    def __init__(self, **kwargs):
        algorithm = GroupLassoBlockAlgorithm(**kwargs)
        super(self.__class__, self).__init__(algorithm, **kwargs)


class GroupLassoBlockAlgorithm(BlockAlgorithm):

    """
    Find sources using group lasso on blocks.

    1) Gaussian group lasso yields a sparse reconstruction of the data.
    2) Features are created using passed function on it to get a single image.
    3) Neural centers are extracted by a local max operation.
    4) Watersheding on non-zero regions yields the returned ROIs.

    Parameters
    ----------
    sig : tuple of ints, shape (D,)
        estimated radius of the neurons in each dimension (e.g., 5 pixels)
    lam : float, optional, default = .5
        Regularization parameter which controls the spatial sparsity
        (initial estimate if targetAreaRatio is non-empty)
    tol : float, optional, default = .01
        Tolerance for stopping FISTA
    maxIter : int, optional, default = 100
        Maximum number of iterations
    nonNegative : boolean, optional, default = True
        If true, neurons should be considered as non-negative
    targetAreaRatio : list, optional, default = []
        If non-empty lam is tuned so that the non-zero area fraction
        (sparsisty) of xk is between targetAreaRatio[0] and targetAreaRatio[1]
    featureCreator : function, optional, default = lambda x: percentile(x, 90)
        Function to compute across images for every pixel to create features,
        e.g. mean, std, max, percentile
    verbose : boolean, optional, default = False
        Print progress if true
    """

    def __init__(self, sig, lam=0.5, tol=1e-2, maxIter=30, nonNegative=True, targetAreaRatio=[],
                 featureCreator=lambda x: percentile(x, 90), verbose=False, **extra):
        self.sig = sig
        self.lam = lam
        self.tol = tol
        self.maxIter = maxIter
        self.nonNegative = nonNegative
        self.targetAreaRatio = targetAreaRatio
        self.featureCreator = featureCreator
        self.verbose = verbose

    def extract(self, block):
        from numpy import zeros, ones, maximum, shape, std, sum, \
            nan_to_num, mean, sqrt, asarray, meshgrid, where, apply_along_axis
        from numpy.linalg import norm
        from scipy.ndimage.filters import gaussian_filter, maximum_filter
        from scipy.ndimage.measurements import label

        def gaussian_group_lasso(data):
            """ Solve gaussian group lasso problem min_x 1/2*||Ax-data||_F^2 + lam*Omega(x)
                where Ax is convolution of x with a Gaussian filter A,
                and Omega is the group l1/l2 norm promoting spatial sparsity
                Input:
                    data : array, shape (T, X, Y[, Z])
                Output:
                    the final value of the iterative minimization
            """
            def A(data, do_transpose=False):
                if type(do_transpose) is bool:
                    # Conveniently, the transpose of a gaussian filter matrix is a
                    # gaussian filter matrix
                    return gaussian_filter(data, (0,) + self.sig, mode='constant')
                elif type(do_transpose) is list:
                    return gaussian_filter(data, tuple(sqrt(len(do_transpose)) * x
                                                       for x in (0,) + self.sig), mode='wrap')
                else:
                    raise NameError(
                        'do_transpose must be bool or list of bools')

            if self.nonNegative:
                prox = lambda x, t: nan_to_num(
                    maximum(1 - t / norm(maximum(x, 0), ord=2, axis=0), 0) * maximum(x, 0))
            else:
                prox = lambda x, t: nan_to_num(
                    maximum(1 - t / norm(x, ord=2, axis=0), 0) * x)

            Omega = lambda x: sum(norm(x, ord=2, axis=0))
            # Lipshitz constant when Gaussian filter is normalized so it sums
            # to 1
            L = 2
            lam = self.lam
            if not self.targetAreaRatio:
                return fista(data, prox, Omega, A, lam, L)
            else:  # Do exponential search to find lam
                lam_high = lam_low = -1
                rho = 10  # exponential search constant
                x = None
                while True:
                    x = fista(data, prox, Omega, A, lam, L, x0=x)
                    y = mean(std(x, 0) > 0)
                    if self.verbose:
                        print(
                            'Area Ratio = {0:.5f},lambda={1:.7f}'.format(y, lam))
                    if y < self.targetAreaRatio[0]:
                        lam_high = lam
                    elif y > self.targetAreaRatio[1]:
                        lam_low = lam
                    else:
                        return x
                    if lam_high == -1:
                        lam *= rho
                    elif lam_low == -1:
                        lam /= rho
                    else:
                        lam = (lam_high + lam_low) / 2

        def fista(data, prox, Omega, A, lam, L, x0=None):
            """ Fast Iterative Soft Threshold Algorithm for solving min_x 1/2*||Ax-data||_F^2 + lam*Omega(x)
                Input:
                    data : array, shape (T, X, Y[, Z])
                    prox : proximal operator of the regularizer Omega
                    Omega: regularizer
                    A    : linear operator applied to x. The named argument 'do_transpose' determines
                           whether to apply the argument or its transpose. If a list of booleans is
                           given, the corresponding operators are applied in order (if possible)
                    lam  : regularization parameter
                    L    : Lipschitz constant. Should be the 2*(the largest eigenvalue of A^T*A).
                  Optional:
                    x0   : Initialization of x
                Output:
                    the final value of the iterative minimization
            """
            tk1 = 1
            sz = data.shape
            if x0 is None:
                x0 = zeros(sz)
            yk = xk = x0
            v = 2 / L * A(data, do_transpose=True)
            for kk in range(self.maxIter):
                xk1 = xk
                tk = tk1
                qk = yk - 2 / L * A(yk, do_transpose=[False, True]) + v
                xk = prox(qk, lam / L)
                tk1 = (1 + sqrt(1 + 4 * (tk ** 2))) / 2
                yk = xk + (tk - 1) / tk1 * (xk - xk1)

                # Adaptive restart from Donoghue2012
                if sum((qk - yk) * (xk - xk1)) > 0:
                    tk1 = tk
                    yk = xk

                norm_xk = norm(xk)
                if norm_xk == 0 or norm(xk - xk1) < self.tol * norm(xk1):
                    return xk

                if self.verbose:
                    resid = A(xk, do_transpose=False) - data
                    resid.shape = (sz[0], -1)
                    loss = norm(resid, ord='fro') ** 2
                    reg = Omega(xk)
                    print('{0:1d}: Obj = {1:.1f}, Loss = {2:.1f}, Reg = {3:.1f}, Norm = {4:.1f}'.format(
                        kk, 0.5 * loss + lam * reg, loss, reg, norm_xk))
            return xk

        def getROI(pic, cent):
            # find ROIs (regions of interest) for image 'pic' given centers -  by
            # choosing all the non-zero points nearest to each center
            dims = shape(pic)
            components, _ = label(pic > 0, ones([3] * len(dims)))
            mesh = meshgrid(indexing='ij', *map(range, dims))
            distances = asarray(
                [sqrt(sum((mesh[i] - c[i]) ** 2 for i in range(len(dims)))) for c in cent])
            min_dist_ind = distances.argmin(0)
            ROI = [asarray(where((components == components[tuple(c)]) *
                                 (min_dist_ind == min_dist_ind[tuple(c)]))).T for c in cent]
            return ROI

        def getCenters(pic):
            # apply the local maximum filter; all pixel of maximal value
            # in their neighborhood are set to 1
            local_max = maximum_filter(
                pic, footprint=ones([3] * pic.ndim)) == pic
            # local_max is a mask containing the peaks looked for, but also the background.
            # To isolate the peaks we must remove the background from the mask.
            local_max[pic == 0] = 0
            return asarray(where(local_max)).T

        x = gaussian_group_lasso(block)
        pic_x = apply_along_axis(self.featureCreator, 0, x)
        # centers extracted from fista output using LocalMax
        cent = getCenters(pic_x)
        # ROI around each center, using watersheding on non-zero regions
        return map(Source, getROI(pic_x,  cent))
