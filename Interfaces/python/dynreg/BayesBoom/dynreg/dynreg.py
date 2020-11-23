import BayesBoom.boom as boom
import numpy as np
import patsy
import BayesBoom.spikeslab as spikeslab
import BayesBoom.R as R
import scipy.sparse


class SparseDynamicRegressionModel:
    """
    A DynamicRegressionModel is a time series regression model where the
    coefficients obey a classic state space model.  Note that the number of
    observations at each time point might differ.  The model is implemented as
    a multivariate state space model.  Through data augmentation one can extend
    this model to most GLM's.

    Define the set of responses at time t as Y'_t = [y_1t, y_2t, ... y_n_tt],
    where Y_t = X[t] * beta[t] + error[t], with temporally IID error term
    error[t] ~ N(0, Diagonal(sigma^2)).

    Each coefficient beta[j, t] is zero with probability determined by a Markov
    chain Pr(beta[j, t] = s | beta[j, t-1] = r) = q[r, s], for r, s in {0, 1}.

    The conditional distribution of beta[j, t] given beta[j, t-1], and given
    that both are nonzero, is normal with mean b_jt = T_ij b_jt-1, and variance
    tau^2.

    Expected Use:

    model = SparseDynamicRegressionModel(3)

    """

    def __init__(self,
                 formula,
                 data,
                 innovation_precision_priors,
                 prior_inclusion_probabilities,
                 expected_inclusion_duration,
                 transition_probability_prior_sample_size,
                 seed=None):
        """
        Args:
          xdim:  The number of potential predictor variables available.

          residual_precision_prior: Prior distribution on the residual
            precision parameter 1/sigsq.  This must be an object inheriting
            from boom.GammaModelBase.  The typcial choice is a boom.ChisqModel.

          innovation_sd_prior_guess:
          innovation_sd_prior_sampler_size:
            Vector-like objects of length xdim that define the prior
            distribution on the innovation parameters tau^2.

          prior_inclusion_probabilities:
          expected_inclusion_duration:
          transition_probability_prior_sample_size:

            These collectively define the prior over the Markov chains
            describing the time series of inclusion indicators for the dynamic
            regression coefficients.  Each is an array-like object containing
            xdim elements.  prior_inclusion_probabilities give the station
        """

        self._model = boom.DynamicRegressionModel(xdim)
        self._create_posterior_sampler(
            residual_precision_prior,
            prior_inclusion_probabilities,
            expected_inclusion_duration,
            transition_probability_prior_sample_size)

        if seed is not None:
            np.random.seed(seed)
            boom.GlobalRng.rng.seed(seed)

    def set_prior(self):
        pass

    def set_data(self, formula, data, timestamps):
        """
        Partitiion the DataFrame 'data' into chunks defined by 'timestamps',
        pass it through the 'formula', and convert the output to the
        expected BayesBoom objects.

        Args:

          formula: A string defining a formula as interpreted by the 'patsy'
            library.

          data: A data frame containing the variables in 'formula', and
            maybe other variables as well.  Extraneous variables will be
            ignored.

          timestamps: A vector-like object containing objects that can be
            ordered.  E.g. a vector of dates, or integers.  Each element of
            'timestamps' corresponds to a row of 'data', and determines
            which the time point to which that row belongs.
        """

        unique_time_points = sorted(set(timestamps))
        responses, predictors = patsy.dmatrices(formula, data, eval_env=1)
        for time_stamp in unique_time_points:
            subset = timestamps == time_stamp
            response, predictors = pasty.dmatrices(formula, data, eval_env=1)


    def train(self, formula, data, timestamps, niter: int, ping: int = None):
        """
        Train a bsts model by running a specified number of MCMC iterations.

        Args:
          formula: A string giving a formula that can be interpreted by the
            'patsy' package (python's version of R's model syntax).

          data: A sequence of DataFrame containing the variables from the
            formula.

          niter:  The desired number of MCMC iterations.
          ping: The frequency with which to print status updates in the MCMC
            algorithm.  The default is niter/10.  If ping <= 0 then no status
            updates are printed.

        Effects:
          The model is populated with MCMC draws.  The regression parameters,
          if any, are stored in the model object itself.  Parameters for any
          state models are stored in the state model objects.  The
          contributions of each state model are stored in the state model
          objects.
        """
        self._set_data(formula, data)
        self._allocate_space(niter)

        for i in range(niter):
            self._model.sample_posterior()
            self._record_state(i)

    def _set_data(self, formula, data):
        #######
        ####### TODO: expose RegressionDataTimePoint.
        #######
        for i in range(len(data)):
            response, predictors = patsy.dmatrices(formula, data)

    def _record_state(self, i):
        ##### TODO: all_coefficients
        self._beta_draws[i, :, :] = self._model.all_coefficients()
        self._residual_sd_draws[i] = self._model.residual_sd
        self._innovation_sd_draws[i, :] = self._model.innovation_sds
        self._transition_probabilities[i, :, :, :] = (
            self._model.transition_probability_matrices()
        )

    @property
    def xdim(self):
        """
        The number of potential predictor variables.
        """
        return self._model.xdim

    @property
    def time_dimension(self):
        """
        The number of time points in the model's data.
        """
        return self._model.time_dimension

    def _allocate_space(self, niter):
        """
        Create space to store 'niter' MCMC draws.
        """
        self._beta_draws = np.zeros((niter, self.xdim, self.time_dimension))
        self._residual_sd_draws = np.zeros((niter))
        self._innovation_sd_draws = np.zeros((niter, self.xdim))
        self._transition_probabilities = np.zeros((niter, self.xdim, 2, 2))
