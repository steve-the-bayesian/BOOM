from .bsts import ObservationModelManager
import patsy
import numpy as np
import BayesBoom.boom as boom
import BayesBoom.R as R
import BayesBoom.spikeslab as spikeslab
import scipy


class StateSpaceStudentModelFactory:
    def __init__(self, formula: str):
        """
        Create a StateSpaceStudentModelFactory.

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

    def create_model(self, prior, data, rng, **kwargs):
        """
        Create the boom model object, and store related model artifacts.

        Args:
          prior: The prior for the observation model.  Either None or a
            spikeslab.StudentSpikeSlabPrior.
          data: If self._formula is None then data is the time series to be
            modeled.  Either a pd.Series or a np.ndarray.  Otherwise data
            should be a pd.DataFrame containing the variables referenced in
            'formula'.
          **kwargs: If prior is None then any remaining arguments are passed to
            the StudentSpikeSlabPrior constructor.

        Returns:
          The created model.

        Effects:
          self._model is populated with the created model.
          self._prior is populated with the prior distribution for the
            observation model.
        """

        if data is not None:
            if self._formula is None:
                # Pure time series case.
                response = data
                predictors = np.ones((len(response), 1))
                kwargs["expected_model_size"] = 0
            else:
                # Time series regression case.
                response, predictors = patsy.dmatrices(self._formula, data)
                self.predictor_names = predictors.design_info.term_names

            boom_response = boom.Vector(R.to_numpy(response))
            boom_predictors = boom.Matrix(R.to_numpy(predictors))
            response_is_observed = np.isfinite(response).ravel()

            self._model = boom.StateSpaceStudentRegressionModel(
                boom_response, boom_predictors, response_is_observed)
        elif prior is not None:
            xdim = len(prior._prior_inclusion_probabilities)
            self._model = boom.StateSpaceStudentRegressionModel(xdim)
            response = None
            predictors = None
        else:
            raise Exception("At least one of 'data' or 'prior' is needed.")

        regression = self._model.observation_model

        prior = self._verify_prior(prior, response, predictors, **kwargs)

        self._prior = prior

        observation_model_sampler = boom.TRegressionSpikeSlabSampler(
            regression,
            prior.slab(regression.Sigsq_prm),
            prior.spike,
            prior.residual_precision,
            prior.tail_thickness,
            rng,
        )
        observation_model_sampler.set_sigma_upper_limit(prior.sigma_upper_limit)
        if prior.max_flips > 0:
            observation_model_sampler.limit_model_selection(prior.max_flips)
        regression.set_method(observation_model_sampler)

        sampler = boom.StateSpaceStudentPosteriorSampler(
            self._model,
            observation_model_sampler)
        self._model.set_method(sampler)

        self._original_series = response
        return self._model

    def create_observation_model_manager(self):
        return StudentObservationModelManager(
            self._model.observation_model.xdim,
            self._formula)

    def _verify_prior(self, prior, response, predictors, **kwargs):
        """
        Ensure that the prior is set up right.
        Args:
          prior: Either None, or a spikeslab.StudentSpikeSlabPrior object.  If
            None then a default prior will be created from the supplied data.
          response: If prior is None, this must either be a vector containing
            the response variable.  Acceptable types include a boom.Vector,
            pd.Series, or a np.ndarray.  If prior is None then this value is
            not used.
          predictors: If prior is None, this must be a Matrix containing the
            predictor values.  Acceptable types include a boom.Matrix,
            pd.DataFrame with all numeric dtypes, or a np.ndarray.
          **kwargs: If prior is None then any additional named arguments are
            passed to the StudentSpikeSlabPrior constructor.

        Returns:
          The validated prior.  This is 'prior' if not None and of the correct
          type.  Otherwise it is the default prior constructed from the
          remaining arguments.

        Exceptions:
          An exception is thrown if 'prior' is not None and it is not a
          StudentSpikeSlabPrior.
        """
        if prior is None:
            prior = spikeslab.StudentSpikeSlabPrior(
                predictors,
                response,
                **kwargs,
            )

        if not isinstance(prior, spikeslab.StudentSpikeSlabPrior):
            raise Exception("Expected a StudentSpikeSlabPrior prior "
                            "distribution.")
        return prior


class StudentObservationModelManager(ObservationModelManager):
    def __init__(self, xdim: int, formula: str):
        """
        In a pure time series context, xdim == 0.  In a regression context,
        xdim is the number of predictors in the regression portion of the model.
        """
        self._xdim = xdim
        self._formula = formula

    @property
    def has_regression(self):
        return self._xdim > 1 and self._formula is not None

    def allocate_space(self, niter: int, time_dimension: int):
        self._residual_sd = np.empty(niter)
        self._residual_df = np.empty(niter)
        if self._xdim > 0:
            self._coefficients = scipy.sparse.lil_matrix((niter, self._xdim))

        if self.has_regression:
            self._regression_contribution = np.empty((niter, time_dimension))

    def record_draw(self, iteration: int, model):
        student_reg = model.observation_model
        self._residual_sd[iteration] = student_reg.residual_sd
        self._residual_df[iteration] = student_reg.residual_df
        if self._xdim > 0:
            self._coefficients[iteration, :] = spikeslab.sparsify(
                student_reg.coef)

        if self.has_regression:
            self._regression_contribution[iteration, :] = (
                model.regression_contribution.to_numpy()
            )

    def restore_draw(self, iteration: int, model):
        student_reg = model.observation_model
        student_reg.set_residual_sd(self._residual_sd[iteration])
        student_reg.set_residual_df(self._residual_df[iteration])
        if self._xdim > 0:
            spikeslab.set_glm_coefs(
                student_reg.coef,
                self._coefficients[iteration, :])

    def format_prediction_data(self, prediction_data, **kwargs):
        if isinstance(prediction_data, int):
            formatted = {
                "forecast_horizon": int(prediction_data),
                "predictors": boom.Matrix(np.ones((int(prediction_data), 1)))
            }
        else:
            predictor_matrix = patsy.dmatrix(
                self._formula, data=prediction_data)
            xnames = predictor_matrix.design_info.term_names
            formatted = {
                "forecast_horizon": prediction_data.shape[0],
                "predictors": boom.Matrix(predictor_matrix),
                "xnames": xnames,
            }
        return formatted

    def predict(self, model, formatted_prediction_data, boom_final_state, rng,
                separate_components=False, **kwargs):
        """
        Return one draw from the posterior predictive distribution.
        """
        if separate_components:
            draw = model.simulate_forecast_components(
                rng, formatted_prediction_data["predictors"], boom_final_state)
        else:
            draw = model.simulate_forecast(
                rng, formatted_prediction_data["predictors"], boom_final_state)
        return draw.to_numpy()
