from __future__ import division
from thunder.extraction.block.base import BlockMethod, BlockAlgorithm
from thunder.utils.common import smallestFloatType, checkParams


class BlockSparseGaussian(BlockMethod):

    def __init__(self, **kwargs):
        algorithm = SparseGaussianBlockAlgorithm(**kwargs)
        super(self.__class__, self).__init__(algorithm, **kwargs)


class SparseGaussianBlockAlgorithm(BlockAlgorithm):

    """
    Find sources using Gaussian group lasso on blocks.

    1) Gaussian group lasso yields a sparse deconvolution of the data.
    2) Single image is created computing percentile/mean/stdev across images for every pixel
    3) Centers are extracted by a local max operation.
    4) Optional watersheding on non-zero regions yields ROIs.

    Parameters
    ----------
    sig : tuple of ints, shape (D,)
        Estimated radius of the neurons in each dimension (e.g., 5 pixels)

    lam : float, optional, default = 1.
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

    method : str, optional, default = 'percentile'
            Method to create single image, options are 'percentile', 'mean', 'stdev'

    perc : int, optional, default = 95
        Percentile used to obtain single image

    minDistance : int, optional, default = 2
        Minimum distance between source centers

    adaptBackground : boolean, optional, default = True
        If true, esimate background as rank 1 NMF

    getROI : boolean, optional, default = False
        If true, obtain regions of interest using watersheding

    verbose : boolean, optional, default = False
        Print progress if true

    registration : boolean, optional, default = False
        Blockwise registration using crosscorrelation if true

    """

    def __init__(self, sig, lam=1., tol=1e-2, maxIter=100, nonNegative=True, targetAreaRatio=[],
                 method='percentile', perc=95, minDistance=2, adaptBackground=True,
                 getROI=False, verbose=False, registration=False, **extra):
        checkParams(method, ['mean', 'stdev', 'percentile'])
        self.sig = sig
        self.lam = lam
        self.tol = tol
        self.maxIter = maxIter
        self.nonNegative = nonNegative
        self.targetAreaRatio = targetAreaRatio
        self.method = method.lower()
        self.perc = perc
        self.minDistance = minDistance
        self.adaptBackground = adaptBackground
        self.getROI = getROI
        self.verbose = verbose
        self.registration = registration

    def extract(self, block):
        from numpy import zeros, ones, maximum, shape, std, sum, percentile, dot, outer,\
            nan_to_num, mean, sqrt, asarray, meshgrid, where, argsort, zeros_like
        from numpy.linalg import norm
        from thunder.extraction.source import Source

        def gaussianGroupLasso(data):
            """
            Solve gaussian group lasso problem min_x 1/2*||Ax-data||_F^2 + lam*Omega(x)
            where Ax is convolution of x with a Gaussian filter A,
            and Omega is the group l1/l2 norm promoting spatial sparsity

            Parameters
            ----------
            data : array, shape (T, X, Y[, Z])

            Returns
            -------
            array, shape (T, X, Y[, Z]), the final value x of the iterative minimization
            """
            def A(data, transpose=False):
                from scipy.ndimage.filters import gaussian_filter
                # The named argument 'transpose' determines whether to apply
                # the argument or its transpose. If a list of booleans is given,
                # the corresponding operators are applied in order (if possible).
                # Conveniently, the transpose of a gaussian filter matrix is a
                # gaussian filter matrix
                if type(transpose) is bool:
                    return gaussian_filter(data, (0,) + self.sig)
                elif type(transpose) is list:
                    return gaussian_filter(data, tuple(sqrt(len(transpose)) * x
                                                       for x in (0,) + self.sig))
                else:
                    raise TypeError('transpose must be bool or list of bools')

            if self.nonNegative:
                def prox(x, t):
                    tmp = nan_to_num(  # faster and more memory efficent than numpy.linalg.norm
                        maximum(1 - t / sqrt(sum((maximum(xx, 0) ** 2 for xx in x), axis=0)), 0))
                    q = zeros_like(x)
                    # faster and more memory efficent than q=tmp*maximum(x, 0)
                    for j, xx in enumerate(x):
                        q[j] = tmp * maximum(xx, 0)
                    return q
            else:
                def prox(x, t):
                    tmp = nan_to_num(  # faster and more memory efficent than np.linalg.norm
                        maximum(1 - t / sqrt(sum((xx ** 2 for xx in x), axis=0)), 0))
                    q = zeros_like(x)
                    for j, xx in enumerate(x):  # faster and more memory efficent than q=tmp*x
                        q[j] = tmp * xx
                    return q

            Omega = lambda x: sum(norm(x, ord=2, axis=0))
            # Lipshitz constant when Gaussian filter is normalized so it sums to 1
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
                            'Area Ratio = {0:.5f}, lambda={1:.7f}'.format(y, lam))
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
            """
            Solve min_x 1/2*||Ax-data||_F^2 + lam*Omega(x) using Fast Iterative Soft Threshold Algorithm

            Parameters
            ----------
            data : array, shape (T, X, Y[, Z])

            prox : function
                proximal operator of the regularizer Omega

            Omega : function
                Regularizer

            A : function
                Linear operator applied to x.

            lam : scalar
                Regularization parameter

            L : scalar
                Lipschitz constant. Should be 2*(the largest eigenvalue of A^T*A).

            x0 : array, shape (T, X, Y[, Z]), optional, default = None
                Initialization of x

            Returns
            -------
            array, shape (T, X, Y[, Z]), the final value x of the iterative minimization
            """
            tk1 = 1
            sz = data.shape
            if x0 is None:
                x0 = zeros(sz, dtype=smallestFloatType(data.dtype))
            yk = xk = x0
            del x0
            v = 2 / L * A(data.astype(smallestFloatType(data.dtype)), transpose=True)
            if self.adaptBackground:
                b_t, b_s = greedyNNPCA(
                    data, percentile(data, 30, 0).reshape(-1).astype(smallestFloatType(data.dtype)), 3)
            for kk in range(self.maxIter):
                xk1 = xk
                tk = tk1
                if self.adaptBackground:
                    r = A(yk, transpose=False)
                    qk = - 2 / L * A(r + outer(b_t, b_s).reshape(sz), transpose=True) + v
                    if kk % 5 == 4:
                        b_t, b_s = greedyNNPCA(data - r, b_s, 3)
                else:
                    qk = - 2 / L * A(yk, transpose=[False, True]) + v
                xk = prox(yk+qk, lam / L)
                tk1 = (1 + sqrt(1 + 4 * (tk ** 2))) / 2
                yk = xk + (tk - 1) / tk1 * (xk - xk1)

                # Adaptive restart from Donoghue2012
                if dot(qk.reshape(-1), (xk - xk1).reshape(-1)) > 0:
                    tk1 = tk
                    yk = xk

                norm_xk = norm(xk)
                if norm_xk == 0 or norm(xk - xk1) < self.tol * norm(xk1):
                    return xk

                if self.verbose:
                    resid = A(xk, transpose=False) - data + \
                        (0 if not self.adaptBackground else outer(b_t, b_s).reshape(sz))
                    resid.shape = (sz[0], -1)
                    loss = norm(resid, ord='fro') ** 2
                    reg = Omega(xk)
                    print('{0:1d}: Obj = {1:.1f}, Loss = {2:.1f}, Reg = {3:.1f}, Norm = {4:.1f}'.format(
                        kk, 0.5 * loss + lam * reg, loss, reg, norm_xk))
            return xk

        def greedyNNPCA(data, v_s, iterations):
            # NonNegative greedy PCA
            d = data.reshape(len(data), -1)
            v_s[v_s < 0] = 0
            for _ in range(iterations):
                v_t = dot(d, v_s) / dot(v_s, v_s)
                v_t[v_t < 0] = 0
                v_s = dot(d.T, v_t) / dot(v_t, v_t)
                v_s[v_s < 0] = 0
            return v_t, v_s

        def getROI(im, cent):
            # find regions of interest (ROI) for image 'im' given centers
            # by choosing all the non-zero points nearest to each center
            from scipy.ndimage.measurements import label
            dims = shape(im)
            components, _ = label(im > 0, ones([3] * len(dims)))
            mesh = meshgrid(indexing='ij', *map(range, dims))
            distances = asarray(
                [sqrt(sum((mesh[i] - c[i]) ** 2 for i in range(len(dims)))) for c in cent])
            min_dist_ind = distances.argmin(0)
            ROI = [asarray(where((components == components[tuple(c)]) *
                                 (min_dist_ind == min_dist_ind[tuple(c)]))).T for c in cent]
            return ROI

        def getCenters(im):
            # apply the local maximum filter
            from skimage.feature import peak_local_max
            peaks = peak_local_max(im, min_distance=self.minDistance, threshold_rel=.03).T
            magnitude = im[list(peaks)]
            indices = argsort(magnitude)[::-1]
            peaks = asarray(list(peaks[:, indices]) + [magnitude[indices]]).T
            return peaks

        if self.registration:  # perform registration on block using crosscorrelation
            from thunder.registration.methods.utils import computeDisplacement
            from scipy.ndimage.interpolation import shift
            T = len(block)
            # compute reference image
            if T <= 20:
                ref = block.mean(0)
            else:
                ref = block[int(T / 2) - 10:int(T / 2) + 10].mean(0)
            block = asarray(map(lambda im:
                                shift(im, computeDisplacement(ref, im), mode='nearest'), block))
        x = gaussianGroupLasso(block)
        if x.max() == 0:  # no source in block
            return []
        if self.method == 'mean':
            im_x = mean(x, 0)
        elif self.method == 'stdev':
            im_x = std(x, 0)
        elif self.method == 'percentile':
            im_x = percentile(x, self.perc, 0)
        cent = getCenters(im_x)
        if self.getROI:
            # ROI around each center, using watersheding on non-zero regions
            roi = getROI(im_x,  cent[:, :-1])
            val = map(lambda x: im_x[list(x.T)], roi)
            return [Source(c, v) for (c, v) in zip(roi, val)]
        else:
            return [Source(coordinates=c[:-1], values=c[-1]) for c in cent]
