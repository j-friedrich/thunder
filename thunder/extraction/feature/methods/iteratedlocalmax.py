from thunder.extraction.feature.base import FeatureMethod, FeatureAlgorithm
from thunder.extraction.feature.creators import MaxFeatureCreator
from thunder.extraction.source import SourceModel, Source


class IteratedLocalMax(FeatureMethod):

    def __init__(self, creator=MaxFeatureCreator(), **kwargs):
        algorithm = IteratedLocalMaxFeatureAlgorithm(**kwargs)
        super(self.__class__, self).__init__(algorithm, creator, **kwargs)


class IteratedLocalMaxFeatureAlgorithm(FeatureAlgorithm):
    """
    Find sources by iteratively identifying local maxima in an array.
    Uses rankfiltering to adjust for varying brightness levels.

    Will first find source centers, and if getROI is True automatically
    define a circle around each center using the specified radius

    Parameters
    ----------
    brThresh : int, optional, default = 145
        Brightness threshold to determine regions with cells

    contThresh : int, optional, default = 10
        Local contrast threshold to determine regions with cells

    radius : scalar, optional, default = 4
        Cell radius

    iters : int, optional, default = 2
        Number of iterations

    getROI : boolean, optional, default = False
        If true, obtain regions of interest
    """

    def __init__(self, brThresh=145, contThresh=12, radius=4, iters=2,
                 getROI=False, **extra):
        self.radius = radius
        self.bThresh = brThresh
        self.cThresh = contThresh
        self.iters = iters
        self.getROI = getROI

    def extract(self, im):
        from numpy import pi, argsort, sort, ones, zeros_like, asarray, concatenate
        from numpy.random import randn
        from scipy.ndimage.filters import percentile_filter, gaussian_filter
        from skimage.morphology import disk, binary_opening
        from skimage.filters import rank
        from skimage.feature import peak_local_max
        from skimage.draw import circle

        def foo(im):
            cImg = binary_opening(im - percentile_filter(im, 0, 32) > self.cThresh, disk(2))
            cImg *= (im > self.bThresh)
            rankImg = rank.equalize(im.astype('uint16'), disk(self.radius * 2)) * cImg
            peaks = []
            for i in range(self.iters):
                aveImg = gaussian_filter(rankImg, self.radius / 2. + .5)
                maxImg = peak_local_max(aveImg, min_distance=self.radius + 2, indices=False)
                maxImg[maxImg > 0] += 1e-6 * randn(maxImg.sum())
                peaks += list(peak_local_max(maxImg * im, min_distance=self.radius + 2))
                if i < self.iters - 1:
                    tmp = zeros_like(im)
                    for p in peaks:
                        tmp[tuple(p)] = 1
                    rankImg *= (gaussian_filter(1. * tmp, self.radius + 1)
                                * 2 * pi * (self.radius + 1)**2 < .5)
            return peaks
            peaks = asarray(peaks, dtype='uint16')
            return peaks[argsort([im[tuple(p)] for p in peaks])[::-1]]

        # extract local peaks
        if im.ndim == 2:
            peaks = foo(im)
        else:
            peaks = []
            for i in range(0, im.shape[2]):
                tmp = foo(im[:, :, i])
                peaks = peaks.append(concatenate((tmp, ones((len(tmp), 1)) * i), axis=1))

        # sort peaks
        peaks = asarray(peaks, dtype='uint16')
        val = [im[tuple(p)] for p in peaks]
        peaks = peaks[argsort(val)[::-1]]

        if self.getROI:
            # construct circular regions from peak points
            def pointToCircle(center, radius):
                rr, cc = circle(center[0], center[1], radius)
                return asarray(zip(rr, cc))

            # return circles as sources
            circles = [pointToCircle(p, self.radius) for p in peaks]
            return SourceModel([Source(c) for c in circles])
        else:
            return SourceModel([Source(c, v) for (c, v) in zip(peaks, sort(val)[::-1])])
