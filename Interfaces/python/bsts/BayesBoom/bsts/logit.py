from .bsts import ObservationModelManager
import patsy
import numpy as np
import BayesBoom.boom as boom
import BayesBoom.spikeslab as spikeslab
import scipy
from numbers import Number


class StateSpaceLogitModelFactory:
    def __init__(self, formula):
        """
        Args:
          formula: A string describing how the predictor variables are to be
            constructed from a data frame.  For example "y ~ x1 + x2 + x2*x3".
            If the formula is None then the model will contain no regression
            component.
        """
        if formula is not None and not isinstance(formula, str):
            raise Exception("formula must either be None or a string")
        self._formula = formula
        self.predictor_names = None

    def create_model(self, prior, data, **kwargs):
        if data is not None:
            response, predictors = patsy.dmatrices(self._formula, data)
            self.predictor_names = predictors.design_info.term_names
            extra_args = {**kwargs}
            trials = extra_args.get("trials", 1)
            if isinstance(trials, Number):
                trials = np.full(len(response), trials)
            observed = np.isfinite(response)
            self._model = boom.StateSpacePoissonModel(
                boom.Vector(response),
                boom.Vector(trials),
                boom.Matrix(predictors),
                observed)
        elif prior is not None:
            xdim = len(prior._prior_inclusion_probabilities)
            self._model = boom.StateSpaceLogitModel(xdim)
            response = None
            predictors = None
            trials = None
        else:
            raise Exception("At least one of 'data' or 'prior' is needed.")

        logit_reg = self._model.observation_model
        prior = self._verify_prior(prior, response, predictors, trials,
                                   **kwargs)
        self._prior = prior
        observation_model_sampler = prior.create_sampler(logit_reg, assign=True)

        sampler = boom.StateSpacePoissonPosteriorSampler(
            self._model, observation_model_sampler)
        self._model.set_method(sampler)
        self._original_series = response
        return self._model

    def create_observation_model_manager(self):
        return LogitObservationModelManager(
            xdim=self._model.observation_model.xdim,
            formula=self._formula)

    def _verify_prior(self, prior, response, predictors, trials, **kwargs):
        if prior is None:
            prior = spikeslab.LogitZellnerPrior(
                predictors=predictors,
                response=response,
                trials=trials,
                **kwargs)
        if not isinstance(prior, spikeslab.LogitZellnerPrior):
            raise Exception(
                "Expected 'prior' to be a 'spikeslab.LogitZellnerPrior'."
            )
        return prior


class LogitObservationModelManager(ObservationModelManager):
    def __init__(self, xdim: int, formula: str):
        #  TODO
        self._xdim = xdim
        self._formula = formula

    @property
    def has_regression(self):
        return self._xdim > 1 and self._formula is not None

    @property
    def niter(self):
        if hasattr(self, "_coefficients"):
            return self._coefficients.shape[0]
        else:
            return 0

    def allocate_space(self, niter: int, time_dimension: int):
        self._coefficients = scipy.sparse.lil_matrix((niter, self._xdim))
        if self.has_regression:
            self._regression_contribution = np.empty((niter, time_dimension))

    def record_draw(self, iteration: int, model):
        self._coefficients[iteration, :] = spikeslab.sparsify(
            model.observation_model.coef)
        if self.has_regression:
            self._regression_contribution[iteration, :] = (
                model.regression_contribution.to_numpy()
            )

    def restore_draw(self, iteration: int, model):
        spikeslab.set_glm_coefs(model.observation_model.coef,
                                self._coefficients[iteration, :])

    def format_prediction_data(self, prediction_data, **kwargs):
        if isinstance(prediction_data, int):
            formatted = {
                "forecast_horizon": prediction_data,
                "predictors": boom.Matrix(np.ones((int(prediction_data), 1)))
            }
        else:
            predictor_matrix = patsy.dmatrix(self._formula,
                                             data=prediction_data)
            xnames = predictor_matrix.design_info.term_names
            formatted = {
                "forecast_horizon": prediction_data.shape[0],
                "predictors": boom.Matrix(predictor_matrix),
                "xnames": xnames,
            }
        extra_args = {**kwargs}
        trials = extra_args.get("trials", 1)
        if isinstance(trials, Number):
            trials = np.full(formatted["forecast_horizon"], trials)
        else:
            trials = np.array(trials)
        formatted["trials"] = boom.Vector(trials)

        return formatted

    def predict(self, model, formatted_prediction_data, boom_final_state, rng,
                separate_components=False, **kwargs):
        if separate_components:
            draw = model.simulate_forecast_components(
                rng,
                formatted_prediction_data["predictors"],
                formatted_prediction_data["trials"],
                boom_final_state)
        else:
            draw = model.simulate_forecast(
                rng,
                formatted_prediction_data["predictors"],
                formatted_prediction_data["trials"],
                boom_final_state)
        return draw.to_numpy()
