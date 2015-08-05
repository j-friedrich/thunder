from thunder.rdds.images import Images
from thunder.rdds.imgblocks.blocks import Blocks, PaddedBlocks
from thunder.rdds.timeseries import TimeSeries
from numpy import dot, outer, asarray, argsort


class Demixer(object):

    """
    Demixer of sources using their temporal activities
    """

    def __init__(self, **kwargs):
        self.algorithm = LocalNMFBlockAlgorithm(**kwargs)

    def fit(self, blocks, sources, output='series', size=None, units="pixels", padding=0):
        """
        Fit the demixing model to data

        Parameters
        ----------
        blocks : Blocks, PaddedBlocks, or Images
            Data in blocks, Images will automatically be converted to blocks

        sources : SourceModel
            containing the combined list of sources

        output : string, optional, default = 'series'
            whether to return 'series' or 'blocks'

        size : tuple, optional, default = (50,50,1)
            Block size if converting from images

        units : string, optional, default = "pixels"
            Units for block size specification if converting from images

        padding : int or tuple, optional, default = 0
            Amount of padding if converting from images

        Returns
        -------
        dependent on 'output'
        TimeSeries containing the Sources as keys and their
            activities as values, or
        (Padded)Blocks with tuple (Sources, activities) as values

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
        result = blocks.apply(lambda x: algorithm.demix(x, centers_b,
                                                        False if output == 'series' else True))
        if output == 'series':
            return TimeSeries(result.values().flatMap(lambda x: zip(x[0], x[1])))
        else:
            return result

    def mergeBlocksToTimeSeries(self, fit):
        """
        Convert demixing model from blocks to TimeSeries

        Parameters
        ----------
        fit : Blocks, PaddedBlocks
             Tuples (Sources, activities) obtained from method 'fit'

        Returns
        -------
        TimeSeries containing the Sources as keys and their activities as values
        """
        if isinstance(fit, PaddedBlocks):
            # we remove sources in the padded regions and
            # keep only the ones in the center to avoid duplicates
            # akin to IgnorePaddingBlockMerger without call of collect()
            def removePaddedSources(keySourcesActivity):
                iS = keySourcesActivity[0].imgSlices[1:]
                centers = map(lambda a: a.center, keySourcesActivity[1][0])
                inImgSlice = asarray(map(lambda c: all([i.start <= c[j] <= i.stop - 1
                                                        for j, i in enumerate(iS)]), centers))
                return [keySourcesActivity[1][0][i] for i, k in enumerate(inImgSlice) if k],\
                    keySourcesActivity[1][1][inImgSlice]
            series = fit.rdd.map(removePaddedSources)
        else:
            series = fit.values()
        return TimeSeries(series.flatMap(lambda x: zip(x[0], x[1])))

    def normalizeShape(self, series):
        """
        Normalize shapes such that L2-norm equals 1

        Parameters
        ----------
        series : TimeSeries
            containing the Sources as keys and their activities as values

        Returns
        -------
        TimeSeries containing the Sources as keys and their activities as values
        """
        from numpy.linalg import norm

        def normalize(sa):
            source, activity = sa
            n = norm(source.values)
            source.values /= n
            activity *= n
            return source, activity
        return series.apply(normalize)

    def subtractBackground(self, blocks, fit, size=None, units="pixels", padding=0):
        """
        Subtract background based on rank 1 NMF of the demixing model's residual

        Parameters
        ----------
        blocks : Blocks, PaddedBlocks, or Images
            Data in blocks, Images will automatically be converted to blocks

        fit : Blocks, PaddedBlocks (same as 'blocks')
             Tuples (Sources, activities) obtained from method 'fit'

        size : tuple, optional, default = (50,50,1)
            Block size if converting from images

        units : string, optional, default = "pixels"
            Units for block size specification if converting from images

        padding : int or tuple, optional, default = 0
            Amount of padding if converting from images

        Returns
        -------
        (Padded)Blocks of the data with removed background

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

        res2 = blocks.rdd.zip(fit.values())

        def foo(datafit):
            (key, data), (source, activity) = datafit
            try:  # PaddedBlock
                pIS = key.padImgSlices[1:]
            except:
                pIS = key.imgSlices[1:]
            shift = asarray([p.start for p in pIS])
            denoised_data = 0 * data
            for i, s in enumerate(source):
                for j, c in enumerate(s.coordinates - shift):
                    denoised_data[(slice(0, None),) + tuple(c)] += s.values[j] * activity[i]
            residual = data - denoised_data
            residual.shape = (len(residual), -1)
            b_t = residual.mean(1)
            b_t[b_t < 0] = 0
            for _ in range(5):
                b_s = dot(residual.T, b_t) / dot(b_t, b_t)
                b_s[b_s < 0] = 0
                b_t = dot(residual, b_s) / dot(b_s, b_s)
                b_t[b_t < 0] = 0
            return key, (data - outer(b_t, b_s).reshape(data.shape))

        if isinstance(blocks, PaddedBlocks):
            return PaddedBlocks(res2.map(foo))
        else:
            return Blocks(res2.map(foo))


class LocalNMFBlockAlgorithm(object):

    """
    Demix sources using local NMF.

    Parameters
    ----------
    sig : tuple, shape (D,)
        size of the gaussian kernel in different spatial directions

    nonNegative : boolean, optional, default = True
        if True, sources should be considered as non-negative

    tol : float, optional, default = 1e-6
        tolerance for stopping algorithm

    maxIter : int, optional, default = 100
        maximum number of iterations

    verbose : boolean, optional, default = False
        print progress if true

    registration : boolean, optional, default = False
        Blockwise registration using crosscorrelation if true

    adaptBackground : boolean, optional, default = True
        If true, esimate background as rank 1 NMF

    optimizeCenters : boolean, optional, default = False
        If true, update centers to be center of mass for each source
        and prune wrongly suspected sources
    """

    def __init__(self, sig, nonNegative=True, tol=1e-6, maxIter=100, verbose=False,
                 registration=False, adaptBackground=True, optimizeCenters=False):
        self.sig = asarray(sig)
        self.nonNegative = nonNegative
        self.tol = tol
        self.maxIter = maxIter
        self.verbose = verbose
        self.registration = registration
        self.adaptBackground = adaptBackground
        self.optimizeCenters = optimizeCenters

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
        tuple (key, (sources, activities)), where
            sources : list of Sources, shape(L,)
                the shape vectors
            activity : array, shape (L, T)
                the activity for each source
        """
        from numpy import sum, zeros, ones, asarray, reshape, r_, ix_, exp, arange, dot,\
            outer, where, diff, max, nan_to_num, ravel, vstack, round, any, percentile
        from numpy.linalg import norm
        from scipy.ndimage.measurements import center_of_mass
        from thunder.extraction.source import Source

        # Initialize Parameters
        key = block[0]
        data = block[1]
        if self.registration:
            from thunder.imgprocessing.regmethods.utils import computeDisplacement
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
        centers = map(lambda x: x - [p.start for p in pIS], tmp)
        inImgSlice = map(lambda c: all([i.start <= c[j] + pIS[j].start < i.stop
                                        for j, i in enumerate(iS)]), centers)
        if len(centers) == 0:
            return key, ([], [])
        dims = data.shape
        D = len(dims)
        # size of bounding box is 4 times size of source
        R = 4 * self.sig
        L = len(centers)
        shapes = []
        activity = zeros((L, dims[0]))
        boxes = zeros((L, D - 1, 2), dtype=int)
        MSE_array = []

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
            temp = list(map(lambda a: range(*a), box))
            Z[ix_(*([range(len(Z))] + temp))] += reshape(X, (r_[-1, box[:, 1] - box[:, 0]]))
            return Z

        def regionCut(X, box, *args):
            # Parameters
            #  X : array, shape (T, X, Y[, Z])
            #  box : array, shape (D, 2), region to cut
            #  args : tuple, specificy dimensions of whole picture (optional)
            # Returns
            #  res : array, shape (T, prod(diff(box,1))),
            dims = X.shape
            if len(args) > 0:
                dims = args[0]
            if len(dims) - 1 != len(box):
                raise Exception('box has the wrong number of dimensions')
            return X[ix_(*([list(range(dims[0]))] + list(map(lambda a: range(*a), box))))].reshape((dims[0], -1))

    # Initialize shapes as Gaussians
        for ll in range(L):
            boxes[ll] = getBox(centers[ll], R, dims[1:])
            tmp = [(arange(dims[i + 1]) - centers[ll][i]) ** 2 / (2. * self.sig[i])
                   for i in range(D - 1)]
            tmp = exp(-sum(ix_(*tmp)))
            tmp.shape = (1,) + dims[1:]
            tmp = regionCut(tmp, boxes[ll])
            shapes.append(tmp[0])
        residual = data.copy()
    # Initialize background as 30% percentile
        if self.adaptBackground:
            b_t = ones(len(residual))
            b_s = percentile(residual, 30, 0).reshape(-1)
            residual -= outer(b_t, b_s).reshape(dims)
    # Initialize activity, iteratively remove background
        for _ in range(5):
            # (Re)calculate activity based on data-background and Gaussian shapes
            for ll in range(L):
                X = regionCut(residual, boxes[ll])
                activity[ll] = dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll])
                if self.nonNegative:
                    activity[ll][activity[ll] < 0] = 0
        # (Re)calculate background based on data-sources using nonnegative greedy PCA
            residual = data.copy()
            for ll in range(L):
                residual = regionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])
            if not self.adaptBackground:
                break
            residual.shape = (dims[0], -1)
            b_s = dot(residual.T, b_t) / dot(b_t, b_t)
            b_s[b_s < 0] = 0
            b_t = dot(residual, b_s) / dot(b_s, b_s)
            b_t[b_t < 0] = 0
            residual -= outer(b_t, b_s)
            residual.shape = dims

    # Main Loop
        delete = []
        for kk in range(self.maxIter):
            for ll in range(L):
                if ll in delete:
                    continue
                # Add region
                residual = regionAdd(residual, outer(activity[ll], shapes[ll]), boxes[ll])

                # Cut region
                X = regionCut(residual, boxes[ll])

                # nonnegative greedy PCA
                greedy_pca_iterations = 5
                for _ in range(greedy_pca_iterations):
                    activity[ll] = nan_to_num(dot(X, shapes[ll]) / dot(shapes[ll], shapes[ll]))
                    if self.nonNegative:
                        activity[ll][activity[ll] < 0] = 0

                    shapes[ll] = nan_to_num(
                        dot(X.T, activity[ll]) / dot(activity[ll], activity[ll]))
                    if self.nonNegative:
                        shapes[ll][shapes[ll] < 0] = 0

                if all(shapes[ll] == 0):
                    delete += [ll]
                else:
                    # Subtract region
                    residual = regionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

            # Recalculate background
            if self.adaptBackground:  # and kk % 5 == 0:
                residual = data.copy()
                for ll in range(L):
                    if ll in delete:
                        continue
                    residual = regionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])
                residual.shape = (dims[0], -1)
                for _ in range(greedy_pca_iterations):
                    b_s = dot(residual.T, b_t) / dot(b_t, b_t)
                    b_s[b_s < 0] = 0
                    b_t = dot(residual, b_s) / dot(b_s, b_s)
                    b_t[b_t < 0] = 0
                residual -= outer(b_t, b_s)
                residual.shape = dims

            # Recenter
            if self.optimizeCenters and kk % 30 == 20:
                for ll in range(L):
                    if ll in delete:
                        continue
                    if shapes[ll].max() > .3 * norm(shapes[ll]):  # remove single bright pixel
                        delete += [ll]
                        residual = regionAdd(residual, outer(activity[ll], shapes[ll]), boxes[ll])
                        continue
                    shp = shapes[ll].reshape(ravel(diff(boxes[ll])))
                    com = boxes[ll][:, 0] + round(center_of_mass(shp))
                    # remove if center of mass >3x sigma away
                    if norm(((com - centers[ll])[self.sig > 0]).astype(float) / self.sig[self.sig > 0]) > 3:
                        delete += [ll]
                        residual = regionAdd(residual, outer(activity[ll], shapes[ll]), boxes[ll])
                        continue
                    newbox = getBox(com, R, dims[1:])
                    if any(newbox != boxes[ll]):
                        newshape = zeros(ravel(diff(newbox)))
                        lower = vstack([newbox[:, 0], boxes[ll][:, 0]]).max(0)
                        upper = vstack([newbox[:, 1], boxes[ll][:, 1]]).min(0)
                        newshape[ix_(*map(lambda a: range(*a),
                                          asarray([lower - newbox[:, 0], upper - newbox[:, 0]]).T))] = \
                            shp[ix_(*map(lambda a: range(*a),
                                         asarray([lower - boxes[ll][:, 0], upper - boxes[ll][:, 0]]).T))]
                        residual = regionAdd(residual, outer(activity[ll], shapes[ll]), boxes[ll])
                        shapes[ll] = newshape.reshape(-1)
                        boxes[ll] = newbox
                        residual = regionAdd(residual, -outer(activity[ll], shapes[ll]), boxes[ll])

            # Measure MSE
            MSE = dot(residual.ravel(), residual.ravel())
            if self.verbose:
                print('{0:1d}: MSE = {1:.3f}'.format(kk, MSE))
            if kk > 0 and abs(1 - MSE / MSE_array[-1]) < self.tol:
                break
            if kk == (self.maxIter - 1):
                print('Maximum iteration limit reached')
            MSE_array.append(MSE)

        # change format from shapes and boxes to Sources
        s = []
        a = []
        for ll in range(L):
            if (not inImgSlice[ll] and not returnPadded) or all(shapes[ll] == 0) or ll in delete:
                continue
            coord = asarray(
                where(reshape(shapes[ll], diff(boxes[ll])) > 0)).T + boxes[ll][:, 0]
            coord += asarray([p.start for p in pIS])
            s += [Source(coord, shapes[ll][shapes[ll] > 0])]
            a += [activity[ll]]
        return key, (s, asarray(a))
