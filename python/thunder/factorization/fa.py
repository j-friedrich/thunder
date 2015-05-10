"""
Class for Factor Analysis

requires spark 1.3.1
"""

from thunder.factorization.svd import SVD
from thunder.rdds.series import Series
from thunder.rdds.matrices import RowMatrix
from numpy import log, pi, ones, inf, sqrt, sum, maximum, newaxis,\
    outer, add, subtract, multiply, divide, eye, var
from scipy import linalg


class FA(object):

    """
    Factor analysis on a distributed matrix.

    The observations are assumed to be caused by a linear transformation of
    lower dimensional latent factors and added Gaussian noise.
    Observations can be grouped by rows or columns, indicated by the flag
    'rowFormat', to avoid to transpose the distributed matrix.
    Because the distributed matrix is typically tall and skinny, all matrices
    of size (nrows,_) are of type RowMatrix.

    Parameters
    ----------
    k : int
        Number of factors to estimate

    svdMethod : str, optional, default = "auto"
        If set to 'direct', will compute the SVD with direct gramian matrix estimation and eigenvector decomposition.
        If set to 'em', will approximate the SVD using iterative expectation-maximization algorithm.
        If set to 'auto', will use 'em' if number of columns in input data exceeds 750, otherwise will use 'direct'.

    tol : float, optional, default = 1e-2
        Stopping tolerance for iterative algorithm.

    maxIter : int, optional, default = 1000
        Maximum number of iterations.

    rowFormat : boolean, optional, default = True
        If True then each row is regarded a sample, columns are features
        If False then each column is regarded a sample, rows are features


    Attributes
    ----------
    `comps` : if rowFormat: array, shape (k, ncols)
                      else: RowMatrix, nrows, each of shape (k,)
        The k factor loadings

    `loglike` : float
        The log likelihood.

    `noiseVar` : if rowFormat: array, shape=(ncols,)
                         else: RowMatrix, nrows, each of shape (1,)
        The estimated noise variance for each feature.

    See also
    --------
    SVD : singular value decomposition
    PCA : principal component analysis
    """

    def __init__(self, k=3, svdMethod='auto', tol=1e-2, maxIter=1000, rowFormat=True):
        self.k = k
        self.svdMethod = svdMethod
        self.tol = tol
        self.maxIter = maxIter
        self.rowFormat = rowFormat
        self.svdMaxIter = 20
        self.comps = None
        self.noiseVar = None
        self.loglike = None

    def fit(self, data):
        """
        Fit the FactorAnalysis model to data using iterated SVD

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate the components from, must be a collection of
            key-value pairs where the keys are identifiers and the values are
            one-dimensional arrays

        Returns
        ----------
        self : returns an instance of self.
        """

        if not (isinstance(data, Series)):
            raise Exception(
                'Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not RowMatrix:
            data = data.toRowMatrix()

        SMALL = 1e-12
        # use standard values for SVD, could instead maybe match tolerance of
        # SVD to tol
        svd = SVD(k=self.k, method=self.svdMethod, maxIter=self.svdMaxIter)
        if self.rowFormat:  # following along the lines of scikit learn
            mat = data.center(1).cache()
            n_samples, n_features = mat.nrows, mat.ncols
            llconst = n_features * log(2. * pi) + self.k
            variance = data.variance()
            psi = ones(n_features)
            old_ll = -inf
            for i in xrange(self.maxIter):
                # svd.maxIter = min(2*i+1, self.svdMaxIter) # perfrom less iterations in the inner loop initially
                # SMALL helps numerics
                sqrt_psi = sqrt(psi) + SMALL
                scaledmat = mat.dotDivide(sqrt_psi)
                scaledmat.cache()
                svd.calc(scaledmat)
                s = svd.s ** 2 / n_samples
                unexp_var = scaledmat.variance().sum() - sum(s)
                # Use 'maximum' here to avoid sqrt problems.
                W = sqrt(maximum(s - 1., 0.))[:, newaxis] * svd.v
                W *= sqrt_psi
                # loglikelihood
                ll = llconst + sum(log(s))
                ll += unexp_var + sum(log(psi))
                ll *= -n_samples / 2.
                if (ll - old_ll) < self.tol:
                    break
                old_ll = ll
                psi = maximum(variance - sum(W ** 2, axis=0), SMALL)
            else:
                raise Exception('FactorAnalysis did not converge.' +
                                ' You might want' +
                                ' to increase the number of iterations.')
            self.comps = W
            self.noiseVar = psi
            self.loglike = ll
        else:  # instead of calling above with the transposed matrix,
               # transpose the equations
            mat = data.center(0).cache()
            n_features, n_samples = mat.nrows, mat.ncols
            llconst = n_features * log(2. * pi) + self.k
            variance = mat.rdd.mapValues(var).cache()
            psi = variance.mapValues(lambda x: 1.)
            old_ll = -inf
            for i in xrange(self.maxIter):
                # svd.maxIter = min(2*i+1, self.svdMaxIter) # perfrom less iterations in the inner loop initially
                # SMALL helps numerics
                sqrt_psi = psi.mapValues(lambda x: sqrt(x) + SMALL)
                # implement diag(v)^-1 A as A.zip(v).map(divide)
                scaledmat = mat.rdd.zip(sqrt_psi).map(
                    lambda ((k1, x), (k2, y)): (k1, divide(x, y)))
                scaledmat.cache()
                svd.calc(mat._constructor(scaledmat))
                s = svd.s ** 2 / n_samples
                unexp_var = scaledmat.mapValues(
                    var).values().reduce(add) - sum(s)
                # Use 'maximum' here to avoid sqrt problems.
                W = svd.u.dotTimes(sqrt(maximum(s - 1., 0.)))
                # implement diag(v) A  as A.zip(v).map(multiply)
                W = W.rdd.zip(sqrt_psi).map(
                    lambda ((k1, x), (k2, y)): (k1, multiply(x, y)))
                # loglikelihood
                ll = llconst + sum(log(s))
                ll += unexp_var + psi.mapValues(log).values().reduce(add)
                ll *= -n_samples / 2.
                if (ll - old_ll) < self.tol:
                    break
                old_ll = ll
                psi = variance.zip(W.mapValues(lambda x: sum(x ** 2)))\
                    .map(lambda ((k1, x), (k2, y)): (k1, maximum(subtract(x, y), SMALL)))

            else:
                raise Exception('FactorAnalysis did not converge.' +
                                ' You might want' +
                                ' to increase the number of iterations.')
            self.comps = mat._constructor(W, ncols=self.k).__finalize__(mat)
            self.noiseVar = mat._constructor(psi, ncols=1).__finalize__(mat)
            self.loglike = ll

        return self

    def transform(self, data):
        """
        Apply dimensionality reduction to data using the model.

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate latent variables from, must be a collection of
            key-value pairs where the keys are identifiers and the values
            are one-dimensional arrays

        Returns
        -------
        latents : if rowFormat: RowMatrix, nrows, each of shape (k,)
                          else: array, shape (k, ncols)
            The latent variables of the data.
        """

        if not (isinstance(data, Series)):
            raise Exception(
                'Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not RowMatrix:
            data = data.toRowMatrix()
        if self.rowFormat:
            mat = data.center(1)
            Wpsi = self.comps / self.noiseVar
            cov_z = linalg.inv(eye(self.k) + Wpsi.dot(self.comps.T))
            return mat.times(Wpsi.T).times(cov_z)
        else:
            mat = data.center(0)
            # implement diag(v)^-1 A as A.zip(v).map(divide)
            Wpsi = self.comps.rdd.zip(self.noiseVar.rdd)\
                .map(lambda ((k1, x), (k2, y)): (k1, divide(x, y)))
            # implement A' B as A.zip(B).map(outer).reduce(add)
            # this is faster than tmp = comps.times(mat._constructor(Wpsi))
            tmp = self.comps.rdd.zip(Wpsi).map(
                lambda ((k1, x), (k2, y)): outer(x, y)).reduce(add)
            cov_z = linalg.inv(eye(self.k) + tmp)
            # implement A' B as A.zip(B).map(outer).reduce(add)
            # again faster than calling times on RowMatrix
            return cov_z.dot(Wpsi.zip(mat.rdd).map(
                lambda ((k1, x), (k2, y)): outer(x, y)).reduce(add))
