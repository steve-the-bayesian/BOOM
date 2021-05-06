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
    """Fit a linear model with a spike and slab prior using MCMC.

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
