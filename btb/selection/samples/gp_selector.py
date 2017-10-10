from btb.selection.samples import SampleSelector
from btb.utilities import *
import btb.database as db

import numpy as np
from scipy import linalg, optimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.gaussian_process import regression_models as regression
from sklearn.gaussian_process import correlation_models as correlation

MACHINE_EPSILON = np.finfo(np.double).eps


def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.

    Parameters
    ----------

    X : array_like
        An array with shape (n_samples, n_features)

    Returns
    -------

    D : array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.

    ij : arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) // 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0

    # iterate over samples; for each one, compute l1 distance between it and all
    # samples indexed after it
    for k in range(n_samples - 1):
        # compute start/end indices for the next chunk of dists
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)

        # fill out the right chunk of D with the distances
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij


class GPSelector(SampleSelector):
    def __init__(self, parameters, regr='constant', corr='squared_exponential',
                 beta0=None, storage_mode='full', verbose=False, theta0=1e-1,
                 thetaL=None, thetaU=None, optimizer='fmin_cobyla',
                 random_start=1, normalize=True, nugget=10. * MACHINE_EPSILON,
                 random_state=None)):

        super(GPSelector, self).__init__(parameters)

        self.regr = regr
        self.corr = corr
        self.beta0 = beta0
        self.storage_mode = storage_mode
        self.verbose = verbose
        self.theta0 = theta0
        self.thetaL = thetaL
        self.thetaU = thetaU
        self.normalize = normalize
        self.nugget = nugget
        self.optimizer = optimizer
        self.random_start = random_start
        self.random_state = random_state

    def fit(self, X, y):
        """
        The Gaussian Process model fitting method.

        Parameters
        ----------
        X : double array_like
            An array with shape (n_samples, n_features) with the input at which
            observations were made.

        y : double array_like
            An array with shape (n_samples, ) or shape (n_samples, n_targets)
            with the observations of the output to be predicted.

        Returns
        -------
        gp : self
            A fitted Gaussian Process model object awaiting data to perform
            predictions.
        """
        # Run input checks
        self._check_params()

        self.random_state = check_random_state(self.random_state)

        # Force data to 2D numpy.array
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self.y_ndim_ = y.ndim
        if y.ndim == 1:
            y = y[:, np.newaxis]

        # Check shapes of DOE & observations
        n_samples, n_features = X.shape
        _, n_targets = y.shape

        # Run input checks
        self._check_params(n_samples)

        # Normalize data or don't
        if self.normalize:
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            y_mean = np.mean(y, axis=0)
            y_std = np.std(y, axis=0)
            X_std[X_std == 0.] = 1.
            y_std[y_std == 0.] = 1.
            # center and scale X if necessary
            X = (X - X_mean) / X_std
            y = (y - y_mean) / y_std
        else:
            X_mean = np.zeros(1)
            X_std = np.ones(1)
            y_mean = np.zeros(1)
            y_std = np.ones(1)

        # Calculate matrix of distances D between samples
        D, ij = l1_cross_distances(X)
        if (np.min(np.sum(D, axis=1)) == 0.
                and self.corr != correlation.pure_nugget):
            raise Exception("Multiple input features cannot have the same"
                            " target value.")

        # Regression matrix and parameters
        F = self.regr(X)
        n_samples_F = F.shape[0]
        if F.ndim > 1:
            p = F.shape[1]
        else:
            p = 1
        if n_samples_F != n_samples:
            raise Exception("Number of rows in F and X do not match. Most "
                            "likely something is going wrong with the "
                            "regression model.")
        if p > n_samples_F:
            raise Exception(("Ordinary least squares problem is undetermined "
                             "n_samples=%d must be greater than the "
                             "regression model size p=%d.") % (n_samples, p))
        if self.beta0 is not None:
            if self.beta0.shape[0] != p:
                raise Exception("Shapes of beta0 and F do not match.")

        # Set attributes
        self.X = X
        self.y = y
        self.D = D
        self.ij = ij
        self.F = F
        self.X_mean, self.X_std = X_mean, X_std
        self.y_mean, self.y_std = y_mean, y_std

        # Determine Gaussian Process model parameters
        if self.thetaL is not None and self.thetaU is not None:
            # Maximum Likelihood Estimation of the parameters
            if self.verbose:
                print("Performing Maximum Likelihood Estimation of the "
                      "autocorrelation parameters...")
            self.theta_, self.reduced_likelihood_function_value_, par = \
                self._arg_max_reduced_likelihood_function()
            if np.isinf(self.reduced_likelihood_function_value_):
                raise Exception("Bad parameter region. "
                                "Try increasing upper bound")

        else:
            # Given parameters
            if self.verbose:
                print("Given autocorrelation parameters. "
                      "Computing Gaussian Process model parameters...")
            self.theta_ = self.theta0
            self.reduced_likelihood_function_value_, par = \
                self.reduced_likelihood_function()
            if np.isinf(self.reduced_likelihood_function_value_):
                raise Exception("Bad point. Try increasing theta0.")

        self.beta = par['beta']
        self.gamma = par['gamma']
        self.sigma2 = par['sigma2']
        self.C = par['C']
        self.Ft = par['Ft']
        self.G = par['G']

        return self

    def predict(self, X):
        """
        This function evaluates the Gaussian Process model at x.

        Parameters
        ----------
        X : array_like
            An array with shape (n_eval, n_features) giving the point(s) at
            which the prediction(s) should be made.

        Returns
        -------
        y : array_like, shape (n_samples, ) or (n_samples, n_targets)
            An array with shape (n_eval, ) if the Gaussian Process was trained
            on an array of shape (n_samples, ) or an array with shape
            (n_eval, n_targets) if the Gaussian Process was trained on an array
            of shape (n_samples, n_targets) with the Best Linear Unbiased
            Prediction at x.
        """
        check_is_fitted(self, "X")

        # Check input shapes
        X = check_array(X)
        n_eval, _ = X.shape
        n_samples, n_features = self.X.shape
        n_samples_y, n_targets = self.y.shape

        # Run input checks
        self._check_params(n_samples)

        if X.shape[1] != n_features:
            raise ValueError(("The number of features in X (X.shape[1] = %d) "
                              "should match the number of features used "
                              "for fit() "
                              "which is %d.") % (X.shape[1], n_features))

        # Normalize input
        X = (X - self.X_mean) / self.X_std

        # Initialize output
        y = np.zeros(n_eval)

        # Get pairwise componentwise L1-distances to the input training set
        dx = manhattan_distances(X, Y=self.X, sum_over_features=False)
        # Get regression function and correlation
        f = self.regr(X)
        r = self.corr(self.theta_, dx).reshape(n_eval, n_samples)

        # Scaled predictor
        y_ = np.dot(f, self.beta) + np.dot(r, self.gamma)

        # Predictor
        y = (self.y_mean + self.y_std * y_).reshape(n_eval, n_targets)

        if self.y_ndim_ == 1:
            y = y.ravel()

        return y

    def reduced_likelihood_function(self, theta=None):
        """
        This function determines the BLUP parameters and evaluates the reduced
        likelihood function for the given autocorrelation parameters theta.

        Maximizing this function wrt the autocorrelation parameters theta is
        equivalent to maximizing the likelihood of the assumed joint Gaussian
        distribution of the observations y evaluated onto the design of
        experiments X.

        Parameters
        ----------
        theta : array_like, optional
            An array containing the autocorrelation parameters at which the
            Gaussian Process model parameters should be determined.
            Default uses the built-in autocorrelation parameters
            (ie ``theta = self.theta_``).

        Returns
        -------
        reduced_likelihood_function_value : double
            The value of the reduced likelihood function associated to the
            given autocorrelation parameters theta.

        par : dict
            A dictionary containing the requested Gaussian Process model
            parameters:

            - ``sigma2`` is the Gaussian Process variance.
            - ``beta`` is the generalized least-squares regression weights for
              Universal Kriging or given beta0 for Ordinary Kriging.
            - ``gamma`` is the Gaussian Process weights.
            - ``C`` is the Cholesky decomposition of the correlation
              matrix [R].
            - ``Ft`` is the solution of the linear equation system
              [R] x Ft = F
            - ``G`` is the QR decomposition of the matrix Ft.
        """
        check_is_fitted(self, "X")

        if theta is None:
            # Use built-in autocorrelation parameters
            theta = self.theta_

        # Initialize output
        reduced_likelihood_function_value = - np.inf
        par = {}

        # Retrieve data
        n_samples = self.X.shape[0]
        D = self.D
        ij = self.ij
        F = self.F

        if D is None:
            # Light storage mode (need to recompute D, ij and F)
            D, ij = l1_cross_distances(self.X)
            if (np.min(np.sum(D, axis=1)) == 0.
                    and self.corr != correlation.pure_nugget):
                raise Exception("Multiple X are not allowed")
            F = self.regr(self.X)

        # Set up R
        r = self.corr(theta, D)
        R = np.eye(n_samples) * (1. + self.nugget)
        R[ij[:, 0], ij[:, 1]] = r
        R[ij[:, 1], ij[:, 0]] = r

        # Cholesky decomposition of R
        try:
            C = linalg.cholesky(R, lower=True)
        except linalg.LinAlgError:
            return reduced_likelihood_function_value, par

        # Get generalized least squares solution
        Ft = linalg.solve_triangular(C, F, lower=True)
        Q, G = linalg.qr(Ft, mode='economic')

        sv = linalg.svd(G, compute_uv=False)
        rcondG = sv[-1] / sv[0]
        if rcondG < 1e-10:
            # Check F
            sv = linalg.svd(F, compute_uv=False)
            condF = sv[0] / sv[-1]
            if condF > 1e15:
                raise Exception("F is too ill conditioned. Poor combination "
                                "of regression model and observations.")
            else:
                # Ft is too ill conditioned, get out (try different theta)
                return reduced_likelihood_function_value, par

        Yt = linalg.solve_triangular(C, self.y, lower=True)
        if self.beta0 is None:
            # Universal Kriging
            beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
        else:
            # Ordinary Kriging
            beta = np.array(self.beta0)

        rho = Yt - np.dot(Ft, beta)
        sigma2 = (rho ** 2.).sum(axis=0) / n_samples
        # The determinant of R is equal to the squared product of the diagonal
        # elements of its Cholesky decomposition C
        detR = (np.diag(C) ** (2. / n_samples)).prod()

        # Compute/Organize output
        reduced_likelihood_function_value = - sigma2.sum() * detR
        par['sigma2'] = sigma2 * self.y_std ** 2.
        par['beta'] = beta
        par['gamma'] = linalg.solve_triangular(C.T, rho)
        par['C'] = C
        par['Ft'] = Ft
        par['G'] = G

        return reduced_likelihood_function_value, par

    def _arg_max_reduced_likelihood_function(self):
        """
        This function estimates the autocorrelation parameters theta as the
        maximizer of the reduced likelihood function.
        (Minimization of the opposite reduced likelihood function is used for
        convenience)

        Parameters
        ----------
        self : All parameters are stored in the Gaussian Process model object.

        Returns
        -------
        optimal_theta : array_like
            The best set of autocorrelation parameters (the sought maximizer of
            the reduced likelihood function).

        optimal_reduced_likelihood_function_value : double
            The optimal reduced likelihood function value.

        optimal_par : dict
            The BLUP parameters associated to thetaOpt.
        """

        # Initialize output
        best_optimal_theta = []
        best_optimal_rlf_value = []
        best_optimal_par = []

        if self.verbose:
            print("The chosen optimizer is: " + str(self.optimizer))
            if self.random_start > 1:
                print(str(self.random_start) + " random starts are required.")

        percent_completed = 0.

        # Force optimizer to fmin_cobyla if the model is meant to be isotropic
        if self.optimizer == 'Welch' and self.theta0.size == 1:
            self.optimizer = 'fmin_cobyla'

        if self.optimizer == 'fmin_cobyla':

            def minus_reduced_likelihood_function(log10t):
                return - self.reduced_likelihood_function(
                    theta=10. ** log10t)[0]

            constraints = []
            for i in range(self.theta0.size):
                constraints.append(lambda log10t, i=i:
                                   log10t[i] - np.log10(self.thetaL[0, i]))
                constraints.append(lambda log10t, i=i:
                                   np.log10(self.thetaU[0, i]) - log10t[i])

            for k in range(self.random_start):

                if k == 0:
                    # Use specified starting point as first guess
                    theta0 = self.theta0
                else:
                    # Generate a random starting point log10-uniformly
                    # distributed between bounds
                    log10theta0 = (np.log10(self.thetaL)
                                   + self.random_state.rand(*self.theta0.shape)
                                   * np.log10(self.thetaU / self.thetaL))
                    theta0 = 10. ** log10theta0

                # Run Cobyla
                try:
                    log10_optimal_theta = \
                        optimize.fmin_cobyla(minus_reduced_likelihood_function,
                                             np.log10(theta0).ravel(), constraints,
                                             iprint=0)
                except ValueError as ve:
                    print("Optimization failed. Try increasing the ``nugget``")
                    raise ve

                optimal_theta = 10. ** log10_optimal_theta
                optimal_rlf_value, optimal_par = \
                    self.reduced_likelihood_function(theta=optimal_theta)

                # Compare the new optimizer to the best previous one
                if k > 0:
                    if optimal_rlf_value > best_optimal_rlf_value:
                        best_optimal_rlf_value = optimal_rlf_value
                        best_optimal_par = optimal_par
                        best_optimal_theta = optimal_theta
                else:
                    best_optimal_rlf_value = optimal_rlf_value
                    best_optimal_par = optimal_par
                    best_optimal_theta = optimal_theta
                if self.verbose and self.random_start > 1:
                    if (20 * k) / self.random_start > percent_completed:
                        percent_completed = (20 * k) / self.random_start
                        print("%s completed" % (5 * percent_completed))

            optimal_rlf_value = best_optimal_rlf_value
            optimal_par = best_optimal_par
            optimal_theta = best_optimal_theta

        elif self.optimizer == 'Welch':

            # Backup of the given attributes
            theta0, thetaL, thetaU = self.theta0, self.thetaL, self.thetaU
            corr = self.corr
            verbose = self.verbose

            # This will iterate over fmin_cobyla optimizer
            self.optimizer = 'fmin_cobyla'
            self.verbose = False

            # Initialize under isotropy assumption
            if verbose:
                print("Initialize under isotropy assumption...")
            self.theta0 = check_array(self.theta0.min())
            self.thetaL = check_array(self.thetaL.min())
            self.thetaU = check_array(self.thetaU.max())
            theta_iso, optimal_rlf_value_iso, par_iso = \
                self._arg_max_reduced_likelihood_function()
            optimal_theta = theta_iso + np.zeros(theta0.shape)

            # Iterate over all dimensions of theta allowing for anisotropy
            if verbose:
                print("Now improving allowing for anisotropy...")
            for i in self.random_state.permutation(theta0.size):
                if verbose:
                    print("Proceeding along dimension %d..." % (i + 1))
                self.theta0 = check_array(theta_iso)
                self.thetaL = check_array(thetaL[0, i])
                self.thetaU = check_array(thetaU[0, i])

                def corr_cut(t, d):
                    return corr(check_array(np.hstack([optimal_theta[0][0:i],
                                                       t[0],
                                                       optimal_theta[0][(i +
                                                                         1)::]])),
                                d)

                self.corr = corr_cut
                optimal_theta[0, i], optimal_rlf_value, optimal_par = \
                    self._arg_max_reduced_likelihood_function()

            # Restore the given attributes
            self.theta0, self.thetaL, self.thetaU = theta0, thetaL, thetaU
            self.corr = corr
            self.optimizer = 'Welch'
            self.verbose = verbose

        else:

            raise NotImplementedError("This optimizer ('%s') is not "
                                      "implemented yet. Please contribute!"
                                      % self.optimizer)

        return optimal_theta, optimal_rlf_value, optimal_par

    def _check_params(self, n_samples=None):

        # Check regression model
        if not callable(self.regr):
            if self.regr in self._regression_types:
                self.regr = self._regression_types[self.regr]
            else:
                raise ValueError("regr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._regression_types.keys(), self.regr))

        # Check regression weights if given (Ordinary Kriging)
        if self.beta0 is not None:
            self.beta0 = np.atleast_2d(self.beta0)
            if self.beta0.shape[1] != 1:
                # Force to column vector
                self.beta0 = self.beta0.T

        # Check correlation model
        if not callable(self.corr):
            if self.corr in self._correlation_types:
                self.corr = self._correlation_types[self.corr]
            else:
                raise ValueError("corr should be one of %s or callable, "
                                 "%s was given."
                                 % (self._correlation_types.keys(), self.corr))

        # Check storage mode
        if self.storage_mode != 'full' and self.storage_mode != 'light':
            raise ValueError("Storage mode should either be 'full' or "
                             "'light', %s was given." % self.storage_mode)

        # Check correlation parameters
        self.theta0 = np.atleast_2d(self.theta0)
        lth = self.theta0.size

        if self.thetaL is not None and self.thetaU is not None:
            self.thetaL = np.atleast_2d(self.thetaL)
            self.thetaU = np.atleast_2d(self.thetaU)
            if self.thetaL.size != lth or self.thetaU.size != lth:
                raise ValueError("theta0, thetaL and thetaU must have the "
                                 "same length.")
            if np.any(self.thetaL <= 0) or np.any(self.thetaU < self.thetaL):
                raise ValueError("The bounds must satisfy O < thetaL <= "
                                 "thetaU.")

        elif self.thetaL is None and self.thetaU is None:
            if np.any(self.theta0 <= 0):
                raise ValueError("theta0 must be strictly positive.")

        elif self.thetaL is None or self.thetaU is None:
            raise ValueError("thetaL and thetaU should either be both or "
                             "neither specified.")

        # Force verbose type to bool
        self.verbose = bool(self.verbose)

        # Force normalize type to bool
        self.normalize = bool(self.normalize)

        # Check nugget value
        self.nugget = np.asarray(self.nugget)
        if np.any(self.nugget) < 0.:
            raise ValueError("nugget must be positive or zero.")
        if (n_samples is not None
                and self.nugget.shape not in [(), (n_samples,)]):
            raise ValueError("nugget must be either a scalar "
                             "or array of length n_samples.")

        # Check optimizer
        if self.optimizer not in self._optimizer_types:
            raise ValueError("optimizer should be one of %s"
                             % self._optimizer_types)

        # Force random_start type to int
        self.random_start = int(self.random_start)
