import numpy as np
import pandas as pd
import patsy
import BayesBoom.boom as boom
import BayesBoom.R as R
import scipy.sparse

from .priors import RegressionSpikeSlabPrior


def sparsify(glm_coefs):
    # Convert a boom.GlmCoefs objects to a 1-row sparse matrix.
    inc = glm_coefs.inc.included_positions
    zeros = np.zeros(len(inc))
    return scipy.sparse.csr_matrix(
        (glm_coefs.included_coefficients.to_numpy(),
         (zeros, inc)),
        shape=(1, glm_coefs.inc.nvars_possible))


def set_glm_coefs(glm_coefs: boom.GlmCoefs,
                  sparse_coefs: scipy.sparse.lil_matrix):
    """
    Set the internal values of a boom.GlmCoefs to match a 1-row sparse
    matrix.

    Args:
      glm_coefs:  The boom object to be set.
      sparse_coefs: The value to set.

    """
    nonzero_positions = sparse_coefs.nonzero()[1]
    nonzero_values = np.array(sparse_coefs.data[0])
    glm_coefs.set_sparse_coefficients(nonzero_values, nonzero_positions)


def dot(data_frame, omit=[]):
    """
    Build a formula string by "summing" all entries except those on an 'omit
    list'.  This would typically include the name of the variable on the left
    hand side of the equation.

    This function is named for the 'dot' operator in R, where a formula given
    as 'y ~ .' means "regress y on all other variables".  The R dot operator
    can also be used to specify interactions, as in y ~ .^2.  To allow for
    similar specifications, the return value of this function is wrapped in
    paraentheses "()".

    Args:
      data_frame: The data frame from which to build the equation.  A list or
        array of column names is also acceptable.

      omit: A list of names (strings) to omit.  As a convenience, if a single
        variable is to be omitted a single string can be passed instead of a
        list containing that single string.

    Returns:
      A string containing the list of names in data_frame, separated by '+'.
      The return value begins with '(' and ends with ')' so that y~dot(data,
      omit=["y"])**2 can be used to specify all 2-way interactions.

    Examples:
      formula = "y ~ " + dot(my_data_frame, "y")
      # Returns "y ~ (X1 + X2 + X3 + extraneous_user_id)"

      formula = "y ~ " + dot(my_data_frame, ["y", "extraneous_user_id"])
      # Returns "y ~ (X1 + X2 + X3)"

      formula = f"y ~ {dot(my_data_frame, omit=["y", "extraneous_user_id"])}**2
      # Returns "y ~ (X1 + X2 + X3)**2"

    """

    # Allow 'omit' to be a string, for the common case where there is just one
    # name to omit.
    if isinstance(omit, str):
        omit = [omit]

    vnames = [x for x in data_frame.columns if x not in omit]
    ans = "(" + "+".join(x for x in vnames) + ")"
    return ans


class lm_spike:
    """
    Fit a linear model with a spike and slab prior using MCMC.

    Typical use:

    from boom.spikeslab import lm_spike
    from sklearn.model_selection import train_test_split

    data = pd.read_csv("mydata.csv")
    train_data, pred_data = train_test_split(data, test_size=100)

    model = lm_spike('y ~ .', niter=1000, data=train_data)
    pred = model.predict(pred_data)

    model.plot()
    model.plot("coefficients")
    model.plot("inc")
    model.plot("resid")

    model.summary()
    model.residuals()

    pred.plot()

    """

    def __init__(self,
                 formula: str,
                 niter: int,
                 data: pd.DataFrame,
                 prior: RegressionSpikeSlabPrior = None,
                 ping: int = None,
                 seed: int = None,
                 **kwargs):
        """Create and a model object and run a specified number of MCMC iterations.

        Args:
          formula: A model formula that can be interpreted by the 'patsy'
            module to produce a model matrix from 'data'.
          niter: The desired number of MCMC iterations.
          data: A pd.DataFrame containing the data with which to train the
            model.
          prior: A SpikeSlabPrior object providing the prior distribution over
            the inclusion indicators, the coefficients, and the residual
            variance parameter.
          ping: The frequency (in iterations) with which to print status
            updates.  If ping is None then niter/10 will be assumed.
          seed: The seed for the C++ random number generator, or None.
          **kwargs: Extra argumnts will be passed to SpikeSlabPrior.

        Returns:
          An lm_spike object.

        """

        response, predictors = patsy.dmatrices(formula, data, eval_env=1)
        self._x_design_info = predictors.design_info
        # xdim = predictors.shape[1]
        # sample_size = predictors.shape[0]
        niter = int(niter)
        if niter <= 0:
            raise Exception("niter should be a positive integer.")

        if ping is None:
            ping = int(niter / 10)
        ping = int(ping)

        if seed is not None:
            boom.GlobalRng.rng.seed(int(seed))

        X = boom.Matrix(predictors)
        y = boom.Vector(response)
        nvars = X.ncol

        self._model = boom.RegressionModel(X, y, False)
        if prior is None:
            prior = RegressionSpikeSlabPrior(x=X, y=y, **kwargs)

        sampler = boom.BregVsSampler(
            self._model,
            prior.slab(self._model.Sigsq_prm),
            prior.residual_precision,
            prior.spike)
        self._model.set_method(sampler)
        # A lil matrix is a "linked list" matrix.  This is an efficient method
        # for constructing matrices.  It should be converted to a different
        # matrix type before doing anything with it.
        self._coefficient_draws = scipy.sparse.lil_matrix((niter, nvars))
        self._residual_sd = np.zeros(niter)
        self._log_likelihood = np.zeros(niter)

        for i in range(niter):
            self._model.sample_posterior()
            self._residual_sd[i] = self._model.sigma
            beta = self._model.coef
            self._coefficient_draws[i, :] = sparsify(beta)
            self._log_likelihood[i] = self._model.log_likelihood()

        # Convert the coefficient draws to sparse column format.  Predictions
        # vs this format should take the form X @ beta, not beta @ X.
        self._coefficient_draws = self._coefficient_draws.tocsc()

        self._fitted_values = self.predict(predictors).mean(axis=0)
        self._residuals = y.to_numpy() - self._fitted_values

    @property
    def xdim(self):
        return self._model.xdim

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @property
    def residuals(self):
        return self._residuals

    @property
    def fitted_values(self):
        return self._fitted_values

    @property
    def xnames(self):
        # A list of strings containing the column names of the predictors.
        return self._x_design_info.column_names

    def inclusion_probs(self, burn=None):
        """
        Args:
          burn:  The number of MCMC iterations to discard as burn-in.

        Returns:
          A pd.Series containing the marginal probability that each coefficient
          is nonzero.  The series is indexed by the set of variable names.

        """
        if burn is None:
            burn = R.suggest_burn(self.log_likelihood)
        probs = compute_inclusion_probabilities(
            self._coefficient_draws[burn:, ])
        return pd.Series(probs, index=self.xnames)

    def coefficient_positive_probability(self, burn=None):
        """
        Args:
          burn:  The number of MCMC iterations to discard as burn-in.

        Returns:
          A pd.Series containing the marginal probability that each coefficient
          is positive.

        """
        if burn is None:
            burn = R.suggest_burn(self.log_likelihood)
        probs = coefficient_positive_probability(
            self._coefficient_draws[burn:, :])
        return pd.Series(probs, index=self.xnames)

    def plot(self, what=None, **kwargs):
        """Plot an aspect of the model.

        Args:
          what: The type of plot desired.  Acceptable choices are
            "inclusion", "coefficients", "residual", and "predicted".

          kwargs: Extra arguments are passed to the specific plot function
            being called.

        """

        plot_types = ["inclusion", "coefficients", "residual", "predicted"]
        if what is None:
            what = plot_types[0]
        what = R.unique_match(what, plot_types)
        if what == "coefficients":
            return self.plot_coefficients(**kwargs)
        elif what == "inclusion":
            return self.plot_inclusion(**kwargs)
        elif what == "residual":
            return self.plot_residual(**kwargs)
        elif what == "predicted":
            return self.plot_predicted(**kwargs)
        else:
            raise Exception(f"Unknown plot type {what}.")

    def plot_inclusion(self, burn=None, inclusion_threshold=0,
                       unit_scale=True, number_of_variables=None,
                       ax=None, **kwargs):
        """
        A barplot showing the marginal inclusion probability of each variable.

        Args:
          burn:
        """
        inc = self.inclusion_probs(burn=burn)
        pos = self.coefficient_positive_probability(burn=burn)
        colors = np.array([str(x) for x in pos])
        index = np.argsort(inc.values)[::-1]

        if number_of_variables is None:
            number_of_variables = np.sum(inc >= inclusion_threshold)
        inc = inc[index[:number_of_variables]]
        pos = pos[index[:number_of_variables]]
        colors = colors[index[:number_of_variables]]
        ans = R.barplot(inc,
                        ax=ax,
                        color=colors[::-1],
                        linewidth=.25,
                        edgecolor="black",
                        xlab="Marginal Inclusion Probability",
                        ylab="Variable",
                        **kwargs)
        return ans

    def plot_coefficients(self, **kwargs):
        """A boxplot showing the values of the coefficients.
        """

    def plot_residual(self, hexbin_threshold=1e+5,
                      xlab="fitted", ylab="residual"):
        """A plot of the residuals vs the predicted values.

        """
        fig, ax = R.plot(self.fitted_values, self.residuals,
                         hexbin_threshold=hexbin_threshold,
                         xlab=xlab, ylab=ylab)
        if len(self.residuals) > hexbin_threshold:
            abs_resid = np.abs(self.residuals)
            n = len(abs_resid) - 100
            top_resids = np.argpartition(abs_resid, n)[n:]
            ax.scatter(self.fitted_values[top_resids],
                       self.residuals[top_resids],
                       s=5,
                       color="yellow")
        ax.axhline(color="black", linewidth=.5)
        return fig, ax

    def plot_predicted(self, xlab="fitted", ylab="actual"):
        """A plot of predicted values vs actual values.

        """
        fig, ax = R.plot(self.fitted_values,
                         self.residuals + self.fitted_values,
                         xlab=xlab,
                         ylab=ylab)
        return fig, ax

    def predict(self, newdata, burn=None, seed=None):
        """
        Return an LmSpikePrediciton object.
        """
        if burn is None:
            burn = R.suggest_burn(self.log_likelihood)
        if seed is not None:
            boom.GlobalRng.rng.seed(int(seed))
        if isinstance(newdata, np.ndarray) and len(newdata.shape) == 1:
            newdata = newdata.reshape(1, -1)
        if isinstance(newdata, np.ndarray) and newdata.shape[1] == self.xdim:
            predictors = newdata
        else:
            predictors = patsy.build_design_matrices(
                [self._x_design_info],
                data=newdata)[0]
        return self._coefficient_draws[burn:, :] @ predictors.T

    def summary(self, burn=None):
        """Return a summary of model fit, including something like R^2, and residual
        sd, along with a table of coefficients, standard deviations, and
        marginal inclusion probabilities.

        """
        return lm_spike_summary(self)


class lm_spike_summary:
    """
    Summarizes the fit of an lm_spike model.
    """

    def __repr__(self):
        return """A spike and slab model summary!
        Put R2, residual_sd, and top 10 coefficients here.
        """


def compute_inclusion_probabilities(coefficients):
    """
    Args:
      coefficients:  A (scipy) sparse matrix of regression coefficients.
        Rows represent MCMC draws.  Columns represent variables.

    Returns:
      A np.array of inclusion probabilities.
    """
    nvars = coefficients.shape[1]
    return np.array(
        [
            np.mean(coefficients[:, i] != 0) for i in range(nvars)
        ]
    )


def coefficient_positive_probability(coefficients):
    nvars = coefficients.shape[1]
    return np.array(
        [
            np.mean(coefficients[:, i] > 0) for i in range(nvars)
        ]
    )


def plot_inclusion_probs(coefficients, burn, xnames, inclusion_threshold=0,
                         unit_scale=True, number_of_variables=None, ax=None,
                         **kwargs):
    """
    """
    coef = coefficients[burn:, :]
    inc = compute_inclusion_probabilities(coef)
    pos = coefficient_positive_probability(coef)
    colors = np.array([str(x) for x in pos])
    index = np.argsort(inc.values)[::-1]

    if number_of_variables is None:
        number_of_variables = np.sum(inc >= inclusion_threshold)
    inc = inc[index[:number_of_variables]]
    pos = pos[index[:number_of_variables]]
    colors = colors[index[:number_of_variables]]
    ans = R.barplot(inc,
                    ax=ax,
                    color=colors[::-1],
                    linewidth=.25,
                    edgecolor="black",
                    xlab="Marginal Inclusion Probability",
                    ylab="Variable",
                    **kwargs)
    return ans


def plot_model_size(coefficients, burn, ax=None, **kwargs):
    ndraws = coefficients.shape[0]
    size = np.array([
        np.sum(coefficients[i, :] != 0)
        for i in range(burn, ndraws)
    ])
    return R.hist(size, ax=ax, **kwargs)


class RegressionSlabPrior:
    """
    A multivariate normal distribution intended to be the prior in a multiple
    regression problem.  The prior is

    beta ~ N(b, V)

    where b = (ybar, 0, 0, 0, ....)
    and V^{-1} = kappa * [(1 - alpha) * xtx + alpha * diag(xtx)] / n

    The mean parameter shrinks the intercept towards the sample mean, and all
    other coefficients towards zero.  In the literature it is more standard to
    shrink all coefficients towards zero, but in practice this can inflate
    estimates of the residual standard deviation.

    The prior precision is defined in terms of xtx: the cross product matrix
    from the regression problem.  We average xtx with its diagonal (with weight
    alpha on the diagonal) to ensure that the overall matrix is full rank.  xtx
    is the information matrix for the regression coefficients in a standard
    regression problem, so dividing by 'n' (the sample size) turns the whole
    thing into the "average information from a single observation."
    Multiplying by 'kappa' means that the information content of the prior is
    equivalent to 'kappa' prior observations.
    """

    def __init__(self, xtx, sample_mean, data_sample_size,
                 prior_sample_size=1.0, diagonal_shrinkage=0.05):
        """
        Args:
          Please see the class comments, above.
          xtx:  The cross product matrix from the regression.
          sample_mean: The mean of the response variable in the regression
            problem ('ybar' above).
          data_sample_size: The number of observations in the regression ('n'
            above).
          prior_sample_size: The number of observations of prior weight to
            assign the prior.  ('kappa' above).
          diagonal_shrinkage: The weight to assign the diagonal of xtx in the
            full rank adjustment.  ('alpha' above).
        """
        self._xtx = xtx
        self._sample_mean = sample_mean
        self._data_sample_size = data_sample_size
        self._prior_sample_size = prior_sample_size
        self._diagonal_shrinkage = diagonal_shrinkage

    def set_xtx(self, xtx: np.ndarray):
        self._xtx = xtx

    def boom(self, sigsq_param: boom.UnivParams):
        xtx = self._xtx if self._xtx is not None else np.ones((1, 1))
        return boom.RegressionSlabPrior(
            boom.SpdMatrix(xtx),
            sigsq_param,
            self._sample_mean,
            self._data_sample_size,
            self._prior_sample_size,
            self._diagonal_shrinkage)


class BigAssSpikeSlab:
    """
    A regression trained using a spike and slab regression.  This class differs
    from lm_spike in that we don't expect the full data set to be stored in
    memory.

    The BigAssSpikeSlab requires the data to be streamed through the model
    twice.  The first stream is an initial screen used to identify candidate
    variables.

    Because the model is based on streaming data, encoding data frames into
    numeric matrices of predictors needs to happen outside the model object.
    """

    def __init__(self,
                 xdim: int,
                 subordinate_model_max_dim: int = 500,
                 force_intercept: bool = True,
                 spike=None,
                 slab: RegressionSlabPrior = None,
                 residual_sd_prior: R.SdPrior = None,
                 expected_model_size=1.0,
                 expected_Rsqure=0.5,
                 prior_sample_size=1.0,
                 seed: int = None,
                 **kwargs):
        """
        Args:
          xdim:  The dimension of the large sparse vector of predictors.
          subordinate_model_max_dim: Each subordinate model will manage at most
            this many predictors.
          force_intercept: Should each subordinate model be required to have an
            intercept term?

        """
        self._xdim = xdim
        self._subordinate_model_max_dim = subordinate_model_max_dim
        self._force_intercept = force_intercept

        self._model = boom.BigRegressionModel(
            xdim, subordinate_model_max_dim, force_intercept)

        if slab is None:
            self._slab = RegressionSlabPrior(None, np.nan, -1)
        else:
            self._slab = slab
        assert isinstance(self._slab, RegressionSlabPrior)

        if spike is None:
            self._spike = np.full(xdim, expected_model_size / xdim)
        else:
            self._spike = np.array(spike)

        self._residual_sd_prior = residual_sd_prior
        if residual_sd_prior is not None:
            assert isinstance(residual_sd_prior, R.SdPrior)

        self._response_suf = boom.GaussianSuf()

        self._sampler = None
        self._expected_Rsqure = expected_Rsqure
        self._prior_sample_size = prior_sample_size

    @property
    def xdim(self):
        return self._xdim

    def stream_data_for_initial_screen(self, x: np.ndarray, y: np.ndarray):
        """
        Pass the data to the underlying C++ model object for the purpose of
        running an initial screen.

        Arg:
          x: Matrix of predictor variables.  If an intercept term is desired it
            should be present in the first column.  Any dummy variables and
            basis expansions (e.g. splines) should already be included.
          y:  The response vector.
        """
        for i, yi in enumerate(y):
            data_point = boom.RegressionData(yi, boom.Vector(x[i, :]))
            self._model.stream_data_for_initial_screen(data_point)
            self._response_suf.increment(boom.Vector(y))

    def initial_screen(self,
                       niter=1000,
                       threshold: float = .05,
                       max_candidates_per_model: int = 1000,
                       use_threads: bool = True):
        """
        Run an initial screen to identify candidate variables.  Under the covers
        there are multiple models running independent spike and slab samplers
        to identify sets of candidate variables.

        Args:
          niter: The number of MCMC iterations that should be used in the
            initial screen by each subordinate model.
          threshold: A probability.  Predictors with marginal inclusion
            probabilities above this threshold in their subordinate models will
            be promoted as candidates for inclusion in the second round.
          max_candidates_per_model: If the number of predictor variable
            candidates in a subordinate model exceeds this number then the
            candidate list will be truncated to this number.
          use_threads: If True then the underlying C++ code will run the screen
            in parallel using C++11 threads.  Set this to True unless you've
            got a really good reason not to.
        """
        self._ensure_priors()
        self._sampler.initial_screen(
            niter, threshold, use_threads)

    def _ensure_priors(self):
        if self._residual_sd_prior is None:
            sample_var = self._response_suf.sample_var
            residual_var = (1 - self._expected_Rsqure) * sample_var
            self._residual_sd_prior = R.SdPrior(
                np.sqrt(residual_var), self._prior_sample_size)

        if self._sampler is None:
            self._sampler = boom.BigAssSpikeSlabSampler(
                self._model,
                boom.VariableSelectionPrior(self._spike),
                self._slab.boom(self._model.Sigsq_prm),
                self._residual_sd_prior.boom())
            self._model.set_method(self._sampler)

    def stream_data_for_restricted_model(self, x: np.ndarray, y: np.ndarray):
        """
        After the initial_screen has been run, the data will need to be
        streamed a second time.  The arguments here are identical to
        'stream_data_for_initial_screen'.
        """
        for i, yi in enumerate(y):
            data_point = boom.RegressionData(yi, boom.Vector(x[i, :]))
            self._model.stream_data_for_restricted_model(data_point)

    def train(self, niter):
        """
        This function should only be called after 'initial_screen' and
        'stream_data_for_restricted_model' have completed.

        Run the MCMC algorithm on the potential candidates identified by the
        initial screen.
        """
        self._allocate_space(niter)
        for i in range(niter):
            self._sampler.draw()
            self._residual_sd[i] = self._model.sigma
            beta = self._model.coef
            self._coefficient_draws[i, :] = sparsify(beta)
            self._log_likelihood[i] = self._model.log_likelihood()

    def _allocate_space(self, niter):
        # A lil matrix is a "linked list" matrix.  This is an efficient method
        # for constructing matrices.  It should be converted to a different
        # matrix type before doing anything with it.
        self._coefficient_draws = scipy.sparse.lil_matrix((niter, self.xdim))
        self._residual_sd = np.zeros(niter)
        self._log_likelihood = np.zeros(niter)
