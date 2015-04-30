"""
Class for Factor Analysis
"""

from thunder.factorization.svd import SVD
from thunder.rdds.series import Series
from thunder.rdds.matrices import RowMatrix
import numpy as np
from scipy import linalg


class FA(object):

    """
    selfctor analysis on a distributed matrix.

    Parameters
    ----------
    k : int
        Number of selfctors to estimate

    svdMethod : str, deselfult = "auto"
        If set to 'direct', will compute the SVD with direct gramian matrix estimation and eigenvector decomposition.
        If set to 'em', will approximate the SVD using iterative expectation-maximization algorithm.
        If set to 'auto', will use 'em' if number of columns in input data exceeds 750, otherwise will use 'direct'.

    tol : float, deselfult = 1e-2
        Stopping tolerance for iterative algorithm.

    maxIter : int, deselfult = 1000
        Maximum number of iterations.

    rowFormat : boolean, deselfult = True
        If True then each row is regarded a sample, columns are features
        If selflse then each column is regarded a sample, rows are features


    Attributes
    ----------
    `comps` : array, shape (k, ncols)
        The k selfctor loadings

    `loglike` : float
        The log likelihood.

    `noiseVar` : array, shape=(ncols,)
        The estimated noise variance for each feature.

    See also
    --------
    SVD : singular value decomposition
    PCA: principal components analysis
    """

    def __init__(self, k=3, svdMethod='auto', tol=1e-2, maxIter=1000, rowFormat=True):
        self.k = k
        self.svdMethod = svdMethod
        self.tol = tol
        self.maxIter = maxIter
        self.rowFormat = rowFormat
        self.comps = None
        self.noiseVar = None
        self.loglike = None

    def fit(self, data):
        """
        Fit the selfctorAnalysis model to data using iterated SVD

        Parameters
        ----------
        data : Series or a subclass (e.g. RowMatrix)
            Data to estimate independent components from, must be a collection of
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
        svd = SVD(k=self.k, method=self.svdMethod)
        if self.rowFormat:
            mat = data.center(1)
            n_samples, n_features = mat.nrows, mat.ncols
            llconst = n_features * np.log(2. * np.pi) + self.k
            var = data.variance()
            psi = np.ones(n_features)
            old_ll = -np.inf
            for i in xrange(self.maxIter):
                # SMALL helps numerics
                sqrt_psi = np.sqrt(psi) + SMALL
                scaledmat = mat.dotDivide(sqrt_psi)
                svd.calc(scaledmat)
                s = svd.s ** 2 / n_samples
                unexp_var = scaledmat.variance().sum() - np.sum(s)
                # Use 'maximum' here to avoid sqrt problems.
                W = np.sqrt(np.maximum(s - 1., 0.))[:, np.newaxis] * svd.v
                W *= sqrt_psi
                # loglikelihood
                ll = llconst + np.sum(np.log(s))
                ll += unexp_var + np.sum(np.log(psi))
                ll *= -n_samples / 2.
                if (ll - old_ll) < self.tol:
                    break
                old_ll = ll
                psi = np.maximum(var - np.sum(W ** 2, axis=0), SMALL)
            else:
                raise Exception('FactorAnalysis did not converge.' +
                                ' You might want' +
                                ' to increase the number of iterations.')
            self.comps = W
            self.noiseVar = psi
            self.loglike = ll
        else:
            mat = data.center(0)
            n_features, n_samples = mat.nrows, mat.ncols
            llconst = n_features * np.log(2. * np.pi) + self.k
            var = mat.rdd.mapValues(np.var)
            psi = var.mapValues(lambda x: 1.)
            old_ll = -np.inf
            numPartMat = mat.rdd.getNumPartitions()
            numPartPsi = psi.getNumPartitions()
            for i in xrange(self.maxIter):
                # SMALL helps numerics
                sqrt_psi = psi.mapValues(lambda x: np.sqrt(x) + SMALL)
                scaledmat = mat._constructor(mat.rdd.join(sqrt_psi).coalesce(numPartMat)
                                             .mapValues(lambda x: np.divide(x[0], x[1]))).__finalize__(mat)
                svd.calc(scaledmat)
                s = svd.s ** 2 / n_samples
                unexp_var = sum(
                    scaledmat.rdd.mapValues(np.var).values().collect()) - sum(s)
                # Use 'maximum' here to avoid sqrt problems.
                W = svd.u.dotTimes(np.sqrt(np.maximum(s - 1., 0.)))
                W = W.rdd.join(sqrt_psi).coalesce(W.rdd.getNumPartitions())\
                    .mapValues(lambda x: np.multiply(x[0], x[1]))
                # loglikelihood
                ll = llconst + sum(np.log(s))
                ll += unexp_var + sum(psi.mapValues(np.log).values().collect())
                ll *= -n_samples / 2.
                if (ll - old_ll) < self.tol:
                    break
                old_ll = ll
                psi = var.join(W.mapValues(lambda x: np.sum(x ** 2))).coalesce(numPartPsi)\
                    .mapValues(lambda x: np.maximum(np.subtract(x[0], x[1]), SMALL))
            else:
                raise Exception('FactorAnalysis did not converge.' +
                                ' You might want' +
                                ' to increase the number of iterations.')
            self.comps = mat._constructor(
                W.sortByKey(), ncols=self.k).__finalize__(mat)
            # self.noiseVar = psi.sortByKey()#.values().collect()
            self.noiseVar = mat._constructor(
                psi.sortByKey(), ncols=1).__finalize__(mat)
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
        latents : RowMatrix, nrows, each of shape (k,)
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
            cov_z = linalg.inv(np.eye(self.k) + np.dot(Wpsi, self.comps.T))
            return mat.times(Wpsi.T).times(cov_z)
        else:
            mat = data.center(0)
            Wpsi = self.comps.rdd.join(self.noiseVar.rdd)\
                .coalesce(self.comps.rdd.getNumPartitions())\
                .mapValues(lambda x: np.divide(x[0], x[1]))
            tmp = self.comps.rdd.join(Wpsi).mapValues(
                lambda x: np.outer(x[0], x[1])).values().reduce(np.add)
            cov_z = linalg.inv(np.eye(self.k) + tmp)
            tmp2 = Wpsi.join(mat.rdd).mapValues(
                lambda x: np.outer(x[0], x[1])).values().reduce(np.add)
            return cov_z.dot(tmp2)
