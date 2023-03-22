from numbers import Integral
from numbers import Real
from time import time

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator
from sklearn.manifold import _utils  # type: ignore
from sklearn.metrics.pairwise import _VALID_METRICS
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Hidden
from sklearn.utils._param_validation import Interval
from sklearn.utils._param_validation import StrOptions
from sklearn.utils.validation import check_non_negative

MACHINE_EPSILON = np.finfo(np.double).eps


def _joint_probabilities(distances, desired_perplexity, verbose):
    """Compute joint probabilities p_ij from distances.

    Parameters
    ----------
    distances : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Distances of samples are stored as condensed matrices, i.e.
        we omit the diagonal and duplicate entries and store everything
        in a one-dimensional array.
    desired_perplexity : float
        Desired perplexity of the joint probability distributions.
    verbose : int
        Verbosity level.

    Returns
    -------
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    """
    distances = distances.astype(np.float32, copy=False)
    conditional_P = _utils._binary_search_perplexity(
        distances, desired_perplexity, verbose
    )
    P = conditional_P + conditional_P.T
    sum_P = np.maximum(np.sum(P), MACHINE_EPSILON)
    P = np.maximum(squareform(P) / sum_P, MACHINE_EPSILON)
    return P


def _kl_divergence(
    params,
    P,
    degrees_of_freedom,
    n_samples,
    n_components,
    dist,
    Q,
    compute_error=True,
):
    """t-SNE objective function: gradient of the KL divergence
    of p_ijs and q_ijs and the absolute error.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    dist:
        Distances in target space
    Q : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix in target space.
    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    kl_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij.
    grad : ndarray of shape (n_params,)
        Unraveled gradient of the Kullback-Leibler divergence with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    if compute_error:
        kl_divergence = np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))
    else:
        kl_divergence = np.nan

    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order="K"), X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c

    return kl_divergence, grad


def _jedi_loss(
    params,
    P,
    P_prior,
    degrees_of_freedom,
    n_samples,
    n_components,
    alpha,
    beta,
    compute_error=True,
):
    """JEDI objective function: gradient of the KL divergence
    of p_ijs and q_ijs and Jensen-Shannon divergence of p'_ijs and q_ijs.

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding.
    P : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed joint probability matrix.
    P_prior : ndarray of shape (n_samples * (n_samples-1) / 2,)
        Condensed prior joint probability matrix.
    degrees_of_freedom : int
        Degrees of freedom of the Student's-t distribution.
    n_samples : int
        Number of samples.
    n_components : int
        Dimension of the embedded space.
    alpha: float
        alpha value in Jensen-Shannon divergence
    beta : float
        beta value in Jensen-Shannon divergence
    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    Returns
    -------
    result_divergence : float
        Kullback-Leibler divergence of p_ij and q_ij minus Jensen-Shannon divergence of p'_ijs and q_ijs.
    grad : ndarray of shape (n_params,)
        Unraveled gradient of the objective function with respect to
        the embedding.
    """
    X_embedded = params.reshape(n_samples, n_components)

    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    kl_divergence, grad_kl = _kl_divergence(
        params,
        P,
        degrees_of_freedom,
        n_samples,
        n_components,
        dist,
        Q,
        compute_error,
    )

    PQ = beta * P_prior + (1 - beta) * Q
    QP = beta * Q + (1 - beta) * P_prior

    first_const = P_prior * Q / QP
    first_const = squareform(first_const)
    first_const = np.sum(first_const - np.diag(first_const.diagonal())) * alpha * beta

    tmp = 1 + np.log(Q) - (1 - beta) * Q / PQ - np.log(PQ)

    second_constatnt = Q * tmp
    second_constatnt = squareform(second_constatnt)
    second_constatnt = np.sum(
        second_constatnt - np.diag(second_constatnt.diagonal())
    ) * (1 - alpha)

    first_change = alpha * beta * P_prior / QP

    second_change = -tmp * (1 - alpha)

    sub = np.zeros((X_embedded.shape[0], X_embedded.shape[0], X_embedded.shape[1]))
    for i in range(X_embedded.shape[0]):
        sub[i] = X_embedded[i] - X_embedded

    resulted = (
        dist * Q * (first_change - first_const + second_change + second_constatnt)
    )
    resulted = squareform(resulted)
    resulted = resulted.reshape((resulted.shape[0], resulted.shape[1], 1))
    resulted = sub * resulted

    grad = np.sum(resulted, axis=1)
    grad *= 4

    return kl_divergence, grad.ravel() + grad_kl


def _gradient_descent(
    objective,
    p0,
    it,
    n_iter,
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    args=None,
    kwargs=None,
):
    """Batch gradient descent with momentum and individual gains.

    Parameters
    ----------
    objective : callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.
    p0 : array-like of shape (n_params,)
        Initial parameter vector.
    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).
    n_iter : int
        Maximum number of gradient descent iterations.
    n_iter_check : int, default=1
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization.
    momentum : float within (0.0, 1.0), default=0.8
        The momentum generates a weight for previous gradients that decays
        exponentially.
    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.
    min_gain : float, default=0.01
        Minimum individual gain for each parameter.
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be aborted.
    verbose : int, default=0
        Verbosity level.
    args : sequence, default=None
        Arguments to pass to objective function.
    kwargs : dict, default=None
        Keyword arguments to pass to objective function.

    Returns
    -------
    p : ndarray of shape (n_params,)
        Optimum parameters.
    error : float
        Optimum.
    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    tic = time()
    for i in range(it, n_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        kwargs["compute_error"] = check_convergence or i == n_iter - 1

        error, grad = objective(p, *args, **kwargs)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc
            grad_norm = linalg.norm(grad)

            if verbose >= 2:
                print(
                    "[JEDI] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[JEDI] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[JEDI] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm)
                    )
                break

    return p, error, i


class JEDI(BaseEstimator):
    """JEDI (Jensen Shannon Divergence).
    JEDI is a modification of t-SNE method, which is a tool to visualize
    high-dimensional data. JEDI converts similarities between data points
    to joint probabilities and tries to minimize the Kullback-Leibler
    divergence between the joint probabilities of the low-dimensional embedding
    and the high-dimensional data and maximize Jensen Shannon Divergence
    between joint probabilities of the low-dimensional embedding and prior
    information. JEDI has a cost function that is not convex, i.e. with
    different initializations we can get different results.
    By default, n_components = 2.

    Parameters
    ----------
    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less than the number
        of samples.
    early_exaggeration : float, default=12.0
        Controls how tight natural clusters in the original space are in
        the embedded space and how much space will be between them. For
        larger values, the space between natural clusters will be larger
        in the embedded space. Again, the choice of this parameter is not
        very critical. If the cost function increases during initial
        optimization, the early exaggeration factor or the learning rate
        might be too high.
    learning_rate : float or "auto", default="auto"
        Learning rate for gradient descent.
    n_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.
    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.
    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.
    verbose : int, default=0
        Verbosity level.
    random_state : int, RandomState instance or None, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function.
    alpha : float, default=0.5
        alpha value in Jensen-Shannon divergence
    beta : float, default=0.5
        beta value in Jensen-Shannon divergence

    References
    ----------
    [1] Edith Heiter; Jonas Fischer; Jilles Vreeken;
    FACTORING OUT PRIOR KNOWLEDGE FROM LOW-DIMENSIONAL EMBEDDINGS, arxiv 2103.01828, 2021.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "perplexity": [Interval(Real, 0, None, closed="neither")],
        "early_exaggeration": [Interval(Real, 1, None, closed="left")],
        "learning_rate": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "n_iter": [Interval(Integral, 250, None, closed="left")],
        "n_iter_without_progress": [Interval(Integral, -1, None, closed="left")],
        "min_grad_norm": [Interval(Real, 0, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "init": [
            StrOptions({"pca", "random"}),
            np.ndarray,
        ],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "method": [StrOptions({"barnes_hut", "exact"})],
        "angle": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [None, Integral],
        "square_distances": ["boolean", Hidden(StrOptions({"deprecated"}))],
    }

    # Control the number of exploration iterations with early_exaggeration on
    _EXPLORATION_N_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(
        self,
        perplexity=30.0,
        early_exaggeration=12.0,
        learning_rate="auto",
        n_iter=1000,
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        verbose=0,
        random_state=None,
        alpha=0.5,
        beta=0.5,
    ):
        self.n_components = 2
        self.perplexity = perplexity
        self.early_exaggeration = early_exaggeration
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.init = "random"
        self.verbose = verbose
        self.random_state = random_state
        self.alpha = alpha
        self.beta = beta

    def _check_params_vs_input(self, X):
        if self.perplexity >= X.shape[0]:
            raise ValueError("perplexity must be less than n_samples")

    def _fit(self, X, P_prior):
        """Private function to fit the model using X as training data."""

        self.learning_rate_ = X.shape[0] / self.early_exaggeration / 4
        self.learning_rate_ = np.maximum(self.learning_rate_, 50)

        X = self._validate_data(
            X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
        )

        if X.shape[0] != X.shape[1]:
            raise ValueError("X should be a square distance matrix")

        check_non_negative(
            X,
            "TSNE.fit(). With metric='precomputed', X "
            "should contain positive distances.",
        )

        random_state = check_random_state(self.random_state)

        n_samples = X.shape[0]
        # Retrieve the distance matrix, either using the precomputed one or
        # computing it.
        distances = X

        if np.any(distances < 0):
            raise ValueError(
                "All distances should be positive, the metric given is not correct"
            )

        # compute the joint probability distribution for the input space
        P = _joint_probabilities(distances, self.perplexity, self.verbose)

        assert np.all(np.isfinite(P)), "All probabilities should be finite"
        assert np.all(P >= 0), "All probabilities should be non-negative"
        assert np.all(P <= 1), "All probabilities should be less or then equal to one"

        X_embedded = 1e-4 * random_state.standard_normal(
            size=(n_samples, self.n_components)
        ).astype(np.float32)

        degrees_of_freedom = max(self.n_components - 1, 1)

        return self._jedi(
            P,
            P_prior,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
        )

    def _jedi(
        self,
        P,
        P_prior,
        degrees_of_freedom,
        n_samples,
        X_embedded,
    ):
        """Runs JEDI."""

        params = X_embedded.ravel()

        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate_,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=0),
            "args": [
                P,
                P_prior,
                degrees_of_freedom,
                n_samples,
                self.n_components,
                self.alpha,
                self.beta,
            ],
            "n_iter_without_progress": self._EXPLORATION_N_ITER,
            "n_iter": self._EXPLORATION_N_ITER,
            "momentum": 0.5,
        }

        obj_func = _jedi_loss

        # Learning schedule (part 1): do 250 iteration with lower momentum but
        # higher learning rate controlled via the early exaggeration parameter
        P *= self.early_exaggeration
        P_prior *= self.early_exaggeration
        params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)
        if self.verbose:
            print(
                "[JEDI] KL divergence after %d iterations with early exaggeration: %f"
                % (it + 1, kl_divergence)
            )

        # Learning schedule (part 2): disable early exaggeration and finish
        # optimization with a higher momentum at 0.8
        P /= self.early_exaggeration
        P_prior /= self.early_exaggeration
        remaining = self.n_iter - self._EXPLORATION_N_ITER
        if it < self._EXPLORATION_N_ITER or remaining > 0:
            opt_args["n_iter"] = self.n_iter
            opt_args["it"] = it + 1
            opt_args["momentum"] = 0.8
            opt_args["n_iter_without_progress"] = self.n_iter_without_progress
            params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            print(
                "[JEDI] KL divergence after %d iterations: %f" % (it + 1, kl_divergence)
            )

        X_embedded = params.reshape(n_samples, self.n_components)
        self.kl_divergence_ = kl_divergence

        return X_embedded

    def fit_transform(self, X, P_prior, y=None):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_samples)
            X must be a square distance matrix.
        P_prior: array-like in squareform
            P_prior must be a square distance matrix of prior
            information.
        y : None
            Ignored.

        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        self._validate_params()
        self._check_params_vs_input(X)
        embedding = self._fit(X, P_prior)
        self.embedding_ = embedding
        return self.embedding_

    def fit(self, X, P_prior, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_samples)
            X must be a square distance matrix.
        P_prior: array-like in squareform
            P_prior must be a square distance matrix of prior
            information.
        y : None
            Ignored.

        Returns
        -------
        X_new : array of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        assert len(P_prior.shape) == 1, "P_prior should be in squareform!"
        self._validate_params()
        self.fit_transform(X, P_prior)
        return self

    def _more_tags(self):
        return {"pairwise": True}
