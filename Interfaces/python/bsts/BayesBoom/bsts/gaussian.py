from .bsts import ObservationModelManager
import numpy as np
import pandas as pd
import patsy
import scipy
import BayesBoom.boom as boom
import BayesBoom.R as R
import BayesBoom.spikeslab as spikeslab


class GaussianStateSpaceModelFactory:
    """
    Model factory for models with Gassian observation errors.  Thus the handled
    models could either be boom.StateSpaceModel or
    boom.StateSpaceRegressionModel.
    """
    def __init__(self):
        self._model = None
        self.predictor_names = None

    def create_model(self, prior: R.SdPrior, data: pd.Series, rng):
        """
        Args:
          prior: an R.SdPrior object describing the prior distribution on the
            residual variance paramter.
          data:  The time series of observations as a Pandas Series.
          rng: The boom random number generator.

        Returns:
          A boom.StateSpaceModel object.
        """
        if data is not None:
            if (isinstance(data, np.ndarray)):
                boom_data = boom.Vector(data.ravel().astype(float))
                is_observed = np.isfinite(data)
            else:
                boom_data = boom.Vector(data.values.astype(float))
                is_observed = np.isfinite(data)
            self._model = boom.StateSpaceModel(boom_data, is_observed)
        else:
            self._model = boom.StateSpaceModel()

        if prior is None:
            if data is None:
                raise Exception("One of 'prior' or 'data' must be given")
            sdy = np.std(data, ddof=1)
            prior = R.SdPrior(sigma_guess=sdy, upper_limit=sdy * 1.2)

        if not isinstance(prior, R.SdPrior):
            raise Exception(
                "I expected a prior of type R.SdPrior on the residual variance"
                " parameter."
            )
        self._prior = prior

        boom_prior = boom.ChisqModel(prior.sample_size, prior.sigma_guess)
        observation_model_sampler = boom.ZeroMeanGaussianConjSampler(
            self._model.observation_model,
            boom_prior)
        observation_model_sampler.set_sigma_upper_limit(
            prior.upper_limit)
        self._model.observation_model.set_method(observation_model_sampler)

        sampler = boom.StateSpacePosteriorSampler(self._model, rng)
        self._model.set_method(sampler)

        if data is not None:
            self._original_series = data

        return self._model

    def create_observation_model_manager(self):
        return GaussianObservationModelManager(xdim=0, formula=None)


class StateSpaceRegressionModelFactory:

    def __init__(self, formula):
        if not isinstance(formula, str):
            raise Exception("formula must be a string.")
        self._formula = formula
        self.predictor_names = None

    def create_model(self, prior, data, rng, **kwargs):
        """
        Create the boom model object, and store related model artifacts.

        Args:
          formula:  A model formula describing the regression component.
          data: A pandas DataFrame containing the variables appearing
            'formula'.
          prior: A spikeslab.RegressionSpikeSlabPrior describing the prior
            distribution on the regression coefficients and the residual
            standard deviation.

        Effects: self._model is created, and model formula artifacts are stored
          so they will be available for future predictions.
        """
        response, predictors = patsy.dmatrices(self._formula, data)
        self.predictor_names = predictors.design_info.term_names
        boom_response = boom.Vector(response)
        boom_predictors = boom.Matrix(predictors)
        if prior is None:
            prior = spikeslab.RegressionSpikeSlabPrior(
                boom_predictors, boom_response, **kwargs)

        self._prior = prior

        if not isinstance(prior, spikeslab.RegressionSpikeSlabPrior):
            raise Exception("Unexpected type for prior.")
        is_observed = np.isfinite(response)
        self._model = boom.StateSpaceRegressionModel(
            boom_response, boom_predictors, is_observed)

        reg = self._model.observation_model
        observation_model_sampler = boom.BregVsSampler(
            reg,
            prior.slab(reg.Sigsq_prm),
            prior.residual_precision,
            prior.spike)
        reg.set_method(observation_model_sampler)
        self._original_series = response

        sampler = boom.StateSpacePosteriorSampler(
            self._model)
        self._model.set_method(sampler)

        return self._model

    def create_observation_model_manager(self):
        return GaussianObservationModelManager(self._model.xdim, self._formula)


class GaussianObservationModelManager(ObservationModelManager):
    """
    The observation model manager supports the observation model in the boom
    object. It holds posterior draws of observation model parameters, log
    likelihood, and metadata about the predictors, if any.
    """
    def __init__(self, xdim, formula):
        """
        Args:
          xdim: The dimension of the predictor variable.  For a pure time
            series model with no predictors, call with xdim == 0.
        """
        self._xdim = xdim
        self._formula = formula

    @property
    def has_regression(self):
        return self._xdim > 0

    def allocate_space(self, niter: int, time_dimension: int):
        """
        Create space to hold 'niter' MCMC draws.
        """
        self._residual_sd = np.empty(niter)
        self._log_likelihood = np.empty(niter)

        if self._xdim > 0:
            self._coefficients = scipy.sparse.lil_matrix((niter, self._xdim))

        if self.has_regression > 0:
            self._regression_contribution = np.empty((niter, time_dimension))

    def record_draw(self, iteration: int, model):
        """
        Record the current state of 'model' as of the specified 'iteration'.

        """
        self._residual_sd[iteration] = model.residual_sd
        self._log_likelihood[iteration] = model.log_likelihood
        if self._xdim > 0:
            self._coefficients[iteration, :] = spikeslab.sparsify(model.coef)
            self._regression_contribution[iteration, :] = (
                model.regression_contribution.to_numpy()
            )

    def restore_draw(self, iteration: int, model):
        """
        Set the state of 'model' to the requested iteration.
        """
        model.observation_model.set_sigma(self._residual_sd[iteration])
        if self._xdim > 0:
            model.observation_model.set_sigma(self._residual_sd[iteration])
            spikeslab.set_glm_coefs(model.observation_model.coefs,
                                    self._coefficients[iteration, :])

    def format_prediction_data(self, prediction_data, **kwargs):
        if self._xdim > 0:
            predictor_matrix = patsy.dmatrix(
                self._formula, data=prediction_data)
            formatted = {
                "forecast_horizon": prediction_data.shape[0],
                "predictors": boom.Matrix(predictor_matrix),
            }
        else:
            formatted = {
                "forecast_horizon": int(prediction_data)
            }
        return formatted

    def predict(self, model, formatted_prediction_data, boom_final_state, rng,
                separate_components=False, **kwargs):
        """
        Args:
          model: Either a boom.StateSpaceModel, or a
            boom.StateSpaceRegressionModel.
          formatted_prediction_data: The output of
            self.format_prediction_data().  A dict containing the prediction
            data in the form expected by the boom model.
          burn: The number of MCMC iterations to discard as burn-in.
          **kwargs: Unused.  Extra arguments will cause an exception to be
            raised.
        """
        extra_args = {**kwargs}
        if extra_args:
            raise Exception("Unexpected arguments were given to Bsts.predict:"
                            f"  {extra_args.keys()}.")

        if isinstance(model, boom.StateSpaceRegressionModel):
            if separate_components:
                draw = model.simulate_forecast_components(
                    rng,
                    formatted_prediction_data["predictors"],
                    boom_final_state)
            else:
                draw = model.simulate_forecast(
                    rng,
                    formatted_prediction_data["predictors"],
                    boom_final_state)
        elif isinstance(model, boom.StateSpaceModel):
            if separate_components:
                draw = model.simulate_forecast_components(
                    rng,
                    formatted_prediction_data["forecast_horizon"],
                    boom_final_state)
                # Insert a row of 0's for the regression component.
                draw = R.to_numpy(draw)
                draw = np.insert(draw, -1, np.zeros(draw.shape[1]), axis=0)
            else:
                draw = model.simulate_forecast(
                    rng,
                    formatted_prediction_data["forecast_horizon"],
                    boom_final_state)
        else:
            raise Exception(
                "Unrecognized boom model passed to "
                "GaussianObservationModelManager.predict: "
                f"{model.__class__.__name__}"
            )

        return R.to_numpy(draw)
