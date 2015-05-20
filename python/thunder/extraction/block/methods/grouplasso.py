from __future__ import division

from thunder.extraction.block.base import BlockMethod, BlockAlgorithm
from thunder.extraction.source import Source


class BlockGroupLasso(BlockMethod):

    def __init__(self, **kwargs):
        algorithm = GroupLassoBlockAlgorithm(**kwargs)
        super(self.__class__, self).__init__(algorithm, **kwargs)


class GroupLassoBlockAlgorithm(BlockAlgorithm):

    """

    Parameters
    ----------
    maxIter : int, optional, default = 100
        Maximum number of iterations
    lam  - regularization parameter (initial estimate)
    tol  - tolerance for stopping FISTA
    iters - maximum number of iterations
    nonNegative -  if true, neurons should be considered as non-negative
    targetAreaRatio - if non-empty lamda is tuned so that the non-zero
        area fraction (sparsisty) of xk is between targetAreaRatio[0]
        and targetAreaRatio[1]
    verbose - print progress if true
    """

    def __init__(self, sig, lam=0.5, tol=1e-2, maxIter=30, nonNegative=False,
                 targetAreaRatio=[], verbose=False, **extra):
        self.sig = sig
        self.lam = lam
        self.tol = tol
        self.maxIter = maxIter
        self.nonNegative = nonNegative
        self.targetAreaRatio = targetAreaRatio
        self.verbose = verbose

    def extract(self, block):
        from numpy import zeros, ones, maximum, shape, reshape, prod, std, sum, \
            nan_to_num, mean, percentile, sqrt, array, meshgrid, where, argsort
        from numpy.linalg import norm
        from scipy.ndimage.filters import gaussian_filter, maximum_filter
        from scipy.ndimage.morphology import generate_binary_structure
        from scipy.ndimage.measurements import label

        """ See example script at the bottom.
        Some comments:
        The main function is
        x=gaussian_group_lasso(data,...)
        in which we operate on the data array and extract x,
        which has the same dimensions like data, but is non-zero only
        where neurons are detected. We assume that any "background" has been
        subtracted already (e.g., rank-1 non negative matrix factorization components).

        Later, by doing
        y=std(x,0)
        peaks=detect_peaks(y)
        we output the neuronal centers from x.
        This last bit is somewhat heuristically done using a regional max operation.
        This can tend to over-split neurons. This seems better than using
        connected components, which over-unify neurons (since these neural centers are
        given as initialization to the non-negative matrix factorization script,
        we don't want to be too conservative)

        There are two main parameters:

        1. sig - estimate radius of the neurons (e.g., 5 pixels)
        2. lam - regularization parameter which controls the spatial sparsity.
        Starting from some initial guess, lam can be automatically adjusted
        by setting TargetRange to some range of expected spatial sparsity
        (e.g., [0.05, 0.1] ). When I used this on a large dataset (e.g. whole brain)
        I segement the dataset to patches and, adjust lam on a about 4
        "representative" patches to get a single value which
        gives reasonable performance in all patches. I then used this value of lam
        on the whole brain (just set TargetRange=[] to avoind lam being adjusted).

        One last comment: the algorithm works less well close to the edges
        of the image ("close"~sig). I usually just throw away anything detected
        near the edges and used overlapping patchs to compnesate.

        """

        def gaussian_group_lasso(data):
            """ Solve gaussian group lasso problem min_x 1/2*||Ax-data||_F^2 + lam*Omega(x)
                where Ax is convolution of x with a Gaussian filter A,
                and Omega is the group l1/l2 norm promoting spatial sparsity

                Input:
                    data - Tx(XxYx...Z) array of the data
                Output:
                    xk - the final value of the iterative minimization
            """
            def A(data, do_transpose=False):
                if type(do_transpose) is bool:
                    # Conveniently, the transpose of a gaussian filter matrix is a
                    # gaussian filter matrix
                    return gaussian_filter(data, (0,) + self.sig, mode='constant')
                elif type(do_transpose) is list:
                    return gaussian_filter(data, tuple([sqrt(len(do_transpose)) * x for x in (0,) + self.sig]), mode='wrap')
                else:
                    raise NameError(
                        'do_transpose must be bool or list of bools')

            if self.nonNegative:
                prox = lambda x, t: nan_to_num(
                    maximum(1 - t / norm(maximum(x, 0), ord=2, axis=0), 0) * maximum(x, 0))
            else:
                prox = lambda x, t: nan_to_num(
                    maximum(1 - t / norm(x, ord=2, axis=0), 0) * x)

            Omega = lambda x:   sum(norm(x, ord=2, axis=0))
            # Lipshitz constant when Gaussian filter is normalized so it sums
            # to 1
            L = 2
            lam = self.lam
            if not self.targetAreaRatio:
                return fista(data, prox, Omega, A, lam, L)
            else:  # Do exponential search to find lam
                lam_high = -1
                lam_low = -1
                rho = 10  # exponential search constant
                cond = True
                x = None
                while cond:
                    x = fista(data, prox, Omega, A, lam, L, x0=x)
                    y = mean(std(x, 0) > 0)
                    print('Area Ratio = {0:.5f},lambda={1:.7f}'.format(y, lam))
                    if y < self.targetAreaRatio[0]:
                        lam_high = lam
                    elif y > self.targetAreaRatio[1]:
                        lam_low = lam
                    else:
                        return x
                        cond = False
                    if lam_high == -1:
                        lam = lam * rho
                    elif lam_low == -1:
                        lam = lam / rho
                    else:
                        lam = (lam_high + lam_low) / 2

        def fista(data, prox, Omega, A, lam, L, x0=None):
            """ Fast Iterative Soft Threshold Algorithm for solving min_x 1/2*||Ax-data||_F^2 + lam*Omega(x)
                Input:
                    data - matrix of the data B in the regularized optimization
                    prox - proximal operator of the regularizer Omega
                    Omega - regularizer
                    A    - linear operator applied to x. The named argument 'do_transpose' determines
                           whether to apply the argument or its transpose. If a list of booleans is
                           given, the corresponding operators are applied in order (if possible)
                    lam  - regularization parameter
                    L    - Lipschitz constant. Should be the 2*(the largest eigenvalue of A^T*A).
                  Optional:
                    x0   - Initialization of x
                    tol  - tolerance
                    iters - maximum number of iterations
                    nonNegative - nonNegative -  if true, neurons should be considered
                    as non-negative
                    verbose - print progress if true
                Output:
                    xk - the final value of the iterative minimization 
            """
            tk1 = 1
            if x0 is None:
                x0 = zeros(A(data, do_transpose=True).shape)
            sz = x0.shape
            yk = x0
            xk = x0
            v = 2 / L * A(data, do_transpose=True)
            for kk in range(self.maxIter):
                xk1 = xk
                tk = tk1

                qk = yk - 2 / L * A(yk, do_transpose=[False, True]) + v
                xk = prox(qk, lam / L)

                tk1 = (1 + sqrt(1 + 4 * (tk ** 2))) / 2
                yk = xk + (tk - 1) / tk1 * (xk - xk1)

                # Adaptive restart from Donoghue2012
                do_restart = sum((qk - yk) * (xk - xk1)) > 0
                if do_restart:
                    tk1 = tk
                    yk = xk

                norm_xk = norm(xk)
                if norm_xk == 0:
                    return xk
                elif norm(xk - xk1) / norm(xk1) < self.tol:
                    return xk

                if self.verbose:
                    resid = A(xk, do_transpose=False) - data
                    if len(sz) > 2:
                        resid = reshape(resid, (sz[0], prod(sz[1:])))
                    loss = norm(resid, ord='fro') ** 2
                    reg = Omega(xk)
                    print('{0:1d}: Obj = {1:.1f}, Loss = {2:.1f}, Reg = {3:.1f}, Norm = {4:.1f}'.format(
                        kk, 0.5 * loss + lam * reg, loss, reg, norm_xk))
            return xk

        def GetROI(pic, cent):
            # find ROIs (regions of interest) for image 'pic' given centers -  by
            # choosing all the non-zero points nearest to each center
            BW_locations = pic > 0
            dims = shape(pic)
            components, _ = label(BW_locations, ones([3] * len(dims)))
            # ROI = -ones(dims, dtype=int)
            ROI = []
            mesh = meshgrid(indexing='ij', *map(range, dims))
            distances = array(
                [sqrt(sum((mesh[i] - c[i]) ** 2 for i in range(len(dims)))) for c in cent])
            min_dist_ind = distances.argmin(0)
            for ll in range(len(cent)):
                ind1 = components[tuple(cent[ll])]
                ind2 = min_dist_ind[tuple(cent[ll])]
                comp = (components == ind1)
                comp[min_dist_ind != ind2] = 0
                # ROI[comp] = ll
                ROI += [array(where(comp)).T]
            return ROI

        def GetCenters(image):
            """ Take a image and detect the peaks using local maximum filter.
            input: a 2D image
            output: peaks list, in which
                peaks[0] - y coordinates
                peaks[1] - x coordinates
                peaks[2] - magnitude ("height") of peak
            """
            # define an 8 (26 for 3D)-connected neighborhood
            D = len(image.shape)
            neighborhood = generate_binary_structure(D, D)
            # apply the local maximum filter; all pixel of maximal value
            # in their neighborhood are set to 1
            local_max = maximum_filter(image, footprint=neighborhood) == image
            # local_max is a mask that contains the peaks we are
            # looking for, but also the background.
            # In order to isolate the peaks we must remove the background from the
            # mask.
            ind = image == 0
            local_max[ind] = 0
            peaks = array(where(local_max))
            magnitude = image[list(peaks)]
            indices = argsort(magnitude)[::-1]
            peaks = list(peaks[:, indices]) + [magnitude[indices]]
            return peaks

        x = gaussian_group_lasso(block)
        pic_x = percentile(x, 90, axis=0)
        # centers extracted from fista output using RegionalMax
        cent = GetCenters(pic_x)
        # ROI around each center, using watersheding on non-zero regions
        ROI = GetROI(pic_x,  (array(cent)[:-1]).T)

        return map(Source, ROI)
