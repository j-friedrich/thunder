from thunder.rdds.images import Images
from thunder.rdds.imgblocks.blocks import Blocks, PaddedBlocks
from thunder.rdds.timeseries import TimeSeries
from thunder.utils.common import smallestFloatType
from numpy import asarray, argsort


class Demixer(object):

    """
    Demixer of sources using their temporal activities
    """

    def __init__(self, **kwargs):
        self.algorithm = LocalNMFBlockAlgorithm(**kwargs)

    def fit(self, blocks, sources, size=None, units="pixels", padding=0):
        """
        Fit the demixing model to data

        Parameters
        ----------
        blocks : Blocks, PaddedBlocks, or Images
            Data in blocks, Images will automatically be converted to blocks

        sources : SourceModel
            containing the combined list of sources

        size : tuple, optional, default = (50,50,1)
            Block size if converting from images

        units : string, optional, default = "pixels"
            Units for block size specification if converting from images

        padding : int or tuple, optional, default = 0
            Amount of padding if converting from images

        Returns
        -------
        TimeSeries containing the Sources as keys and their
            activities as values, and optionally
        (Padded)Blocks with tuple (*extras) as values
            if algorithm returns additional output, e.g. the background

        See also
        --------
        Images.toBlocks
        """
        if isinstance(blocks, Images):
            if size is None:
                raise Exception(
                    "Must specify a size if images will be converted to blocks")
            blocks = blocks.toBlocks(size, units, padding)

        elif not (isinstance(blocks, Blocks) or isinstance(blocks, PaddedBlocks)):
            raise Exception("Input must be Images, Blocks, or PaddedBlocks")

        algorithm = self.algorithm
        try:  # sort by values
            centers_b = blocks.rdd.context.broadcast(
                asarray([sources[i].center for i in argsort([s.values.max()
                                                             for s in sources])[::-1]]))
        except AttributeError:  # if sources don't have values
            centers_b = blocks.rdd.context.broadcast(sources.centers)
        result = blocks.apply(lambda x: algorithm.demix(x, centers_b, False)).cache()
        if len(result.values().first()) == 2:
            return TimeSeries(result.values().flatMap(lambda x: zip(x[0], x[1])))
        else:
            return (TimeSeries(result.values().flatMap(lambda x: zip(x[0], x[1]))),
                    result.applyValues(lambda x: x[2:]))


class LocalNMFBlockAlgorithm(object):

    """
    Demix sources using local NMF.

    Parameters
    ----------
    sig : tuple, shape (D,)
        Size of the gaussian kernel in different spatial directions

    nonNegative : boolean, optional, default = True
        If True, sources should be considered as non-negative

    tol : float, optional, default = 1e-6
        Tolerance for stopping algorithm

    iters : int, optional, default = 10
        Number of final iterations on full data

    verbose : boolean, optional, default = False
        Print progress if true

    registration : boolean, optional, default = False
        Blockwise registration using crosscorrelation if true

    adaptBackground : boolean, optional, default = True
        If true, esimate background as rank 1 NMF

    optimizeCenters : boolean, optional, default = False
        If true, update centers to be center of mass for each source

    initIters : list, optional, default = 80
        Numbers of initial iterations on downsampled data

    batchSize : int, optional, default = 30
        Number of frames over which mean is taken in initial iterations

    downSample : tuple, shape (D,), optional, default = None, i.e. no downsampling
        Factor for spatial downsampling in different spatial directions

    thresh : float, optional, default = None, i.e. no merging
        threshold for merging neurons; merge occurs if MSE between the two
        considered NMF-components and the merged one is below threshold
    """

    def __init__(self, sig, nonNegative=True, tol=1e-6, iters=2, verbose=False,
                 registration=False, adaptBackground=True, optimizeCenters=False,
                 initIters=80, batchSize=30, downSample=None, thresh=None):
        self.sig = asarray(sig)
        self.nonNegative = nonNegative
        self.tol = tol
        self.iters = iters
        self.verbose = verbose
        self.registration = registration
        self.adaptBackground = adaptBackground
        self.optimizeCenters = optimizeCenters
        self.initIters = initIters
        self.batchSize = batchSize
        self.downSample = downSample
        self.thresh = thresh

    # main function
    def demix(self, block, allcenters, returnPadded=False):
        """
        Parameters
        ----------
        block : array, shape (T, X, Y[, Z])
            block of the data

        allcenters : array, shape (L, D)
            L centers of suspected sources where D is spatial dimension (2 or 3)

        Returns
        -------
        tuple (key, (sources, activities[, sptlBckgrd, tmprlBckgrd])), where
            sources : list of Sources, shape(L,)
                shape vectors
            activity : array, shape (L, T)
                activity for each source
            sptlBckgrd : array, shape (X, Y[, Z])
                spatial background
            tmprlBckgrd : array, shape (T,)
                temporal background activity

        """
        from numpy import sum, zeros, ones, asarray, reshape, r_, ix_, exp, arange, mean,\
            where, max, nan_to_num, round, any, percentile, sqrt, allclose, repeat, \
            unravel_index, isnan, argmax, unique, hstack, outer, dot, prod
        from numpy.linalg import norm
        from scipy.ndimage.measurements import center_of_mass
        from scipy.signal import welch
        from scipy.ndimage import median_filter
        from thunder.extraction.source import Source

        # Initialize Parameters
        key = block[0]
        data = block[1]
        if self.registration:
            from thunder.registration.methods.utils import computeDisplacement
            from scipy.ndimage.interpolation import shift
            T = len(data)
            # compute reference image
            if T <= 20:
                ref = data.mean(0)
            else:
                ref = data[int(T / 2) - 10:int(T / 2) + 10].mean(0)
            data = asarray(map(lambda im:
                               shift(im, computeDisplacement(ref, im), mode='nearest'), data))
        try:  # PaddedBlock
            pIS = key.padImgSlices[1:]
        except:
            pIS = key.imgSlices[1:]
        iS = key.imgSlices[1:]
        tmp = filter(lambda c: all([p.start <= c[j] < p.stop
                                    for j, p in enumerate(pIS)]), allcenters.value)
        # shift from absolute to box coordinates
        centers = asarray(map(lambda x: x - [p.start for p in pIS], tmp), dtype=int)
        inImgSlice = map(lambda c: all([i.start <= c[j] + pIS[j].start < i.stop
                                        for j, i in enumerate(iS)]), centers)
        if len(centers) == 0:
            return key, ([], [])
        dims = data.shape
        D = len(dims)
        # size of bounding box is 3 times size of source
        R = (3 * self.sig).astype('uint8')
        L = len(centers)
        shapes = []
        mask = []
        boxes = zeros((L, D - 1, 2), dtype='uint16')
        MSE_array = []
        activity = zeros((L, dims[0] / self.batchSize))
        if self.initIters == 0 or self.downSample is None:
            self.downSample = ones(D - 1, dtype='uint8')
        else:
            self.downSample = asarray(self.downSample, dtype='uint8')
        ds = self.downSample

        # Auxiliary functions
        def getBox(centers, R, dims):
            D = len(R)
            box = zeros((D, 2), dtype=int)
            for dd in range(D):
                box[dd, 0] = max((centers[dd] - R[dd], 0))
                box[dd, 1] = min((centers[dd] + R[dd] + 1, dims[dd]))
            return box

        def regionAdd(Z, X, box):
            # Parameters
            #  Z : array, shape (T, X, Y[, Z]), dataset
            #  box : array, shape (D, 2), array defining spatial box to put X in
            #  X : array, shape (T, prod(diff(box,1))), Input
            # Returns
            #  Z : array, shape (T, X, Y[, Z]), Z+X on box region
            Z[[slice(len(Z))] + list(map(lambda a: slice(*a), box))
              ] += reshape(X, (r_[-1, box[:, 1] - box[:, 0]]))
            return Z

        def regionCut(X, box):
            # Parameters
            #  X : array, shape (T, X, Y[, Z])
            #  box : array, shape (D, 2), region to cut
            # Returns
            #  res : array, shape (T, prod(diff(box,1))),
            dims = X.shape
            return X[[slice(dims[0])] + list(map(lambda a: slice(*a), box))].reshape((dims[0], -1))

        def getSnPSD(Y):  # Estimate noise level
            L = len(Y)
            ff, psd_Y = welch(Y, nperseg=round(L / 8))
            sn = sqrt(mean(psd_Y[ff > .3] / 2))
            return sn
        noise = zeros(L)

        def HALS(data, S, activity, skip=[], check_skip=0, iters=1):
            idx = asarray(filter(lambda x: x not in skip, range(len(activity))))
            A = S[idx].dot(data.T)
            B = S[idx].dot(S.T)
            for ii in range(iters):
                for k, ll in enumerate(idx):
                    if check_skip and ii == iters - 1:
                        a0 = activity[ll].copy()
                    activity[ll] += nan_to_num((A[k] - dot(B[k], activity)) / B[k, ll])
                    if self.nonNegative:
                        activity[ll][activity[ll] < 0] = 0
                # skip neurons whose shapes already converged
                    if check_skip and ll < L and ii == iters - 1:
                        if check_skip == 1:  # compute noise level only once
                            noise[ll] = getSnPSD(a0) / a0.mean()
                        if allclose(a0, activity[ll] / activity[ll].mean(), 1e-4, noise[ll]):
                            skip += [ll]
            C = activity[idx].dot(data)
            D = activity[idx].dot(activity.T)
            for _ in range(iters):
                for k, ll in enumerate(idx):
                    if ll == L:
                        S[ll] += nan_to_num((C[k] - dot(D[k], S)) / D[k, ll])
                    else:
                        S[ll, mask[ll]] += nan_to_num((C[k, mask[ll]]
                                                       - dot(D[k], S[:, mask[ll]])) / D[k, ll])
                    if self.nonNegative:
                        S[ll][S[ll] < 0] = 0
            return S, activity, skip

        def HALSactivity(data, S, activity, iters=1):
            A = S.dot(data.T)
            B = S.dot(S.T)
            for _ in range(iters):
                for ll in range(L + self.adaptBackground):
                    activity[ll] += nan_to_num((A[ll] - dot(B[ll].T, activity)) / B[ll, ll])
                    if self.nonNegative:
                        activity[ll][activity[ll] < 0] = 0
            return activity

        def HALSshape(data, S, activity, iters=1):
            C = activity.dot(data)
            D = activity.dot(activity.T)
            for _ in range(iters):
                for ll in range(L + self.adaptBackground):
                    if ll == L:
                        S[ll] += nan_to_num((C[ll] - dot(D[ll], S)) / D[ll, ll])
                    else:
                        S[ll, mask[ll]] += nan_to_num((C[ll, mask[ll]]
                                                       - dot(D[ll], S[:, mask[ll]])) / D[ll, ll])
                    if self.nonNegative:
                        S[ll][S[ll] < 0] = 0
            return S

        def recenter(S, boxes, mask, ds):
            dim = dims[1:] / ds
            for ll in range(len(boxes)):
                com = center_of_mass(S[ll].reshape(dim))
                # com = unravel_index(argmax(median_filter(S[ll].reshape(dim), 3)), dim)
                if isnan(com[0]) or norm((asarray(com) - boxes[ll].mean(1)) * ds / self.sig) > 1:
                    continue
                newbox = getBox(round(com), R / ds, dim)
                if any(newbox != boxes[ll]):
                    temp = zeros(dim)
                    temp[map(lambda a: slice(*a), newbox)] = 1
                    mask[ll] = where(temp.ravel())[0]
                    S[ll] *= temp.ravel()
                    boxes[ll] = newbox
            return S, boxes, mask

        def mergeAll(S, activity, boxes, mask, L, ds):
            dim = dims[1:] / ds

            def merge(S, activity, boxes, mask, i, j, purge):
                # determine merged component
                sCombined = (S[i] / norm(S[i]) + S[j] / norm(S[j]))
                aCombined = ((activity[i] * norm(S[i]) + activity[j] * norm(S[j])) / 2.)
                sa = outer(activity[i], S[i]) + outer(activity[j], S[j])
                for _ in range(3):
                    A = sCombined.dot(sa.T)
                    B = sCombined.dot(sCombined)
                    aCombined = nan_to_num(A / B)
                    if self.nonNegative:
                        aCombined[aCombined < 0] = 0
                    C = aCombined.dot(sa)
                    D = aCombined.dot(aCombined)
                    sCombined = nan_to_num(C / D)
                    if self.nonNegative:
                        sCombined[sCombined < 0] = 0
                shp = sCombined.reshape(dim)
                com = center_of_mass(shp)
                newbox = getBox(round(com), R / ds, dim)
                temp = zeros(dim)
                temp[map(lambda a: slice(*a), newbox)] = 1
                newmask = where(temp.ravel())[0]
            # calc MSE
                qq = 0
                for k in newmask:
                    tmp = aCombined * sCombined[k] - sa[:, k]
                    qq += tmp.dot(tmp)
                for k in filter(lambda a: a not in newmask, unique(hstack([mask[i], mask[j]]))):
                    qq += sa[:, k].dot(sa[:, k])
            # merge only if MSE is smaller than some threshold
                if qq < self.thresh * len(newmask) * len(aCombined):  # * sqrt(sa.mean()):
                    S[i] = sCombined * temp.ravel()
                    boxes[i] = newbox
                    mask[i] = newmask
                    activity[i] = aCombined
                    purge += [j]
                    if self.verbose:
                        print 'merged', i, 'and ', j
                return S, activity, boxes, mask, purge
            purge = []
            com = zeros((L, D - 1))
            for ll in range(L):
                com[ll] = center_of_mass(S[ll].reshape(dim))
                if isnan(com[ll, 0]):
                    purge += [ll]
            # com = boxes.mean(2)
            for l in range(L - 1):
                if l in purge:
                    continue
                for k in range(l + 1, L):
                    if k not in purge and norm((com[l] - com[k]) / asarray(self.sig / ds)) < 2:
                        S, activity, boxes, mask, purge = merge(
                            S, activity, boxes, mask, l, k, purge)
            idx = filter(lambda x: x not in purge, range(L))
            mask = asarray(mask)[idx]
            boxes = asarray(boxes)[idx]
            if self.adaptBackground:
                idx = asarray(idx + [L])
            S = S[idx]
            activity = activity[idx]
            L = len(mask)
            return S, activity, boxes, mask, L

    ### Initialize shapes, activity, and residual ###
        data0 = data[:len(data) / self.batchSize * self.batchSize].reshape((-1, self.batchSize) +
                                                                           data.shape[1:]).mean(1).astype(smallestFloatType(data.dtype))
        if D == 4:
            data0 = data0.reshape(len(data0), dims[1] / ds[0], ds[0],
                                  dims[2] / ds[1], ds[1],
                                  dims[3] / ds[2], ds[2])\
                .mean(2).mean(3).mean(4)
            activity = data0[:, map(int, centers[:, 0] / ds[0]),
                             map(int, centers[:, 1] / ds[1]),
                             map(int, centers[:, 2] / ds[2])].T
        else:
            data0 = data0.reshape(len(data0), dims[1] / ds[0],
                                  ds[0], dims[2] / ds[1],
                                  ds[1]).mean(2).mean(3)
            activity = data0[:, map(int, centers[:, 0] / ds[0]),
                             map(int, centers[:, 1] / ds[1])].T
        dims0 = data0.shape

        data0 = data0.reshape(dims0[0], -1)
        data = data.astype(smallestFloatType(data.dtype)).reshape(dims[0], -1)
        for ll in range(L):
            boxes[ll] = getBox(centers[ll] / ds, R / ds, dims0[1:])
            temp = zeros(dims0[1:])
            temp[map(lambda a: slice(*a), boxes[ll])] = 1
            mask += where(temp.ravel())
            temp = [(arange(dims[i + 1] / ds[i]) - centers[ll][i]
                     / float(ds[i])) ** 2
                    / (2 * (self.sig[i] / float(ds[i])) ** 2)
                    for i in range(D - 1)]
            temp = exp(-sum(ix_(*temp)))
            temp.shape = (1,) + dims0[1:]
            temp = regionCut(temp, boxes[ll])
            shapes.append(temp[0])
        S = zeros((L + self.adaptBackground, prod(dims0[1:])), dtype=smallestFloatType(data.dtype))
        for ll in range(L):
            S[ll] = regionAdd(
                zeros((1,) + dims0[1:]), shapes[ll].reshape(1, -1), boxes[ll]).ravel()
        if self.adaptBackground:
            # Initialize background as 20% percentile
            S[-1] = percentile(data0, 20, 0)
            activity = r_[activity, ones((1, dims0[0]), dtype=smallestFloatType(data.dtype))]

    ### Get shape estimates on subset of data ###
        if self.initIters > 0:
            for kk in range(self.initIters):
                S = HALSshape(data0, S, activity)
                activity = HALSactivity(data0, S, activity)
                if kk > 30:
                    if kk % 20 == 0 and self.optimizeCenters:
                        S, boxes, mask = recenter(S, boxes, mask, ds)
                    if kk % 20 == 5 and self.thresh is not None:
                        S, activity, boxes, mask, L = mergeAll(S, activity, boxes, mask, L, ds)

        ### Back to full data ##
            activity = ones((L + self.adaptBackground, dims[0]),
                            dtype=smallestFloatType(data.dtype)) * activity.mean(1).reshape(-1, 1)
            if D == 4:
                S = repeat(repeat(repeat(S.reshape((-1,) + dims0[1:]), ds[0], 1),
                                  ds[1], 2), ds[2], 3).reshape(L + self.adaptBackground, -1)
            else:
                S = repeat(repeat(S.reshape((-1,) + dims0[1:]),
                                  ds[0], 1), ds[1], 2).reshape(L + self.adaptBackground, -1)
            mask = [[]] * L
            for ll in range(L):
                boxes[ll] *= ds.reshape(-1, 1)
                temp = zeros(dims[1:])
                temp[map(lambda a: slice(*a), boxes[ll])] = 1
                mask[ll] = asarray(where(temp.ravel())[0])

            activity = HALSactivity(data, S, activity, 7)
            S = HALSshape(data, S, activity, 7)

    #### Main Loop ####
        skip = []
        for kk in range(self.iters):
            S, activity, skip = HALS(data, S, activity, skip, iters=10)  # , check_skip=kk)

            # Measure MSE
            if self.verbose:
                residual = data - activity.T.dot(S)
                MSE = dot(residual.ravel(), residual.ravel()) / data.size
                print('{0:1d}: MSE = {1:.5f}'.format(kk, MSE))
                if kk == (self.iters - 1):
                    print('Maximum iteration limit reached')
                MSE_array.append(MSE)

        # change format from shapes and boxes to Sources
        s = []
        a = []
        for ll in range(L):
            if (not inImgSlice[ll] and not returnPadded) or all(shapes[ll] == 0):
                continue
            coord = asarray(where(S[ll].reshape(dims[1:]) > 0)).T
            coord += asarray([p.start for p in pIS])
            n = norm(S[ll])
            s += [Source(coord, S[ll][S[ll] > 0] / n)]
            a += [activity[ll] * n]
        if self.adaptBackground:
            n = S[-1].mean()
            return key, (s, asarray(a), S[-1].reshape(dims[1:]) / n, activity[-1] * n)
        else:
            return key, (s, asarray(a))
