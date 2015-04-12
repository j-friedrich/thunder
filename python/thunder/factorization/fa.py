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
    Factor analysis on a distributed matrix.

    Parameters
    ----------
    k : int
        Number of factors to estimate

    svdMethod : str, optional, default = "auto"
        If set to 'direct', will compute the SVD with direct gramian matrix estimation and eigenvector decomposition.
        If set to 'em', will approximate the SVD using iterative expectation-maximization algorithm.
        If set to 'auto', will use 'em' if number of columns in input data exceeds 750, otherwise will use 'direct'.

    tol : float, default = 1e-2
        Stopping tolerance for iterative algorithm.

    maxIter : int, default = 1000
        Maximum number of iterations.

    Attributes
    ----------
    `comps` : array, shape (k, ncols)
        The k factor loadings

    `loglike` : float
        The log likelihood.

    `noiseVar` : array, shape=(ncols,)
        The estimated noise variance for each feature.

    See also
    --------
    SVD : singular value decomposition
    PCA: principal components analysis
    """

    def __init__(self, k=3, svdMethod='auto', tol=1e-2, maxIter=1000):
        self.k = k
        self.svdMethod = svdMethod
        self.tol = tol
        self.maxIter = maxIter
        self.comps = None
        self.noiseVar = None
        self.loglike = None
        
    def fit(self, data):
        """
        Fit the FactorAnalysis model to data using iterated SVD
        
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
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not RowMatrix:
            data = data.toRowMatrix()

        mat = data.center(1)
        n_samples, n_features = mat.nrows, mat.ncols
        llconst = n_features * np.log(2. * np.pi) + self.k
        var = data.variance()
       
        psi = np.ones(n_features)
        old_ll = -np.inf
        SMALL = 1e-12

        svd = SVD(k=self.k, method=self.svdMethod)
        for i in xrange(self.maxIter):
            # SMALL helps numerics
            sqrt_psi = np.sqrt(psi) + SMALL
            scaledmat=mat._constructor(
                mat.rdd.mapValues(lambda x: np.divide(x, sqrt_psi)),
                nrows=n_samples, ncols=n_features, 
                index=np.arange(n_features)).__finalize__(mat)
            svd.calc(scaledmat)
            s = svd.s**2/n_samples
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
            raise Exception('Input must be Series or a subclass (e.g. RowMatrix)')

        if type(data) is not RowMatrix:
            data = data.toRowMatrix()

        mat = data.center(1)
        Wpsi = self.comps / self.noiseVar
        cov_z = linalg.inv(np.eye(self.k) + np.dot(Wpsi, self.comps.T))
        return mat.times(Wpsi.T).times(cov_z)
