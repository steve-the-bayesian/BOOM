import BayesBoom.boom as boom
import BayesBoom.models as models
import BayesBoom.R as R
import numpy as np
import pandas as pd
import json


from .linear_bandit_encoder import (
    ArmMapJsonEncoder,
    ArmMapJsonDecoder,
    LinearBanditEncoderJSONEncoder,
    LinearBanditEncoderJSONDecoder,
)
    

VALUE_FUNCTION_JSON_ENCODER_REGISTRY = {}
VALUE_FUNCTION_JSON_DECODER_REGISTRY = {}

def register_value_function_json_encoder(
        encoder_name: str,
        json_encoder_class,
        json_decoder_class):
    global VALUE_FUNCTION_JSON_ENCODER_REGISTRY
    global VALUE_FUNCTION_JSON_DECODER_REGISTRY

    if not issubclass(json_encoder_class, json.JSONEncoder):
        raise Exception("Class must be a subclass of json.JSONEncoder")
    if not issubclass(json_decoder_class, json.JSONDecoder):
        raise Exception("Class must be a subclass of json.JSONDecoder")

    VALUE_FUNCTION_JSON_ENCODER_REGISTRY[encoder_name] = json_encoder_class
    VALUE_FUNCTION_JSON_DECODER_REGISTRY[encoder_name] = json_decoder_class


class LogitBandit:
    """
    A multi-armed bandit where the reward function is a logistic regression
    model capable of describing both structured experiments (i.e. multiple
    discrete valued action variables) as well as context (properties of the
    experimental subjects that cannot be changed by the experimenter).
    """

    def __init__(self, arm_map, model_encoder, value_function = None):
        """
        Args:
          arm_map: An ArmMap object describing the arms in the experiment.

          model_encoder: A LinearBanditEncoder object containing all the base
            encoders for the experimental and contextual variables to be
            modeled.
        """
        self._arm_map = arm_map
        self._encoder = model_encoder
        self._value_function = value_function
        self._training_data = []
        self._boom_model = None
        self._boom_sampler = None
        self._boom_bandit = None
        self._prior = None

    @property
    def number_of_arms(self):
        return self._encoder.number_of_arms

    def set_prior(self, prior):
        """
        Set the prior distribution for the model to one of the standard
        priors for binomial logit models.  Accepted families of priors include
        MvnModel, BinomialLogitMvnPrior, and BinomialLogitSpikeSlabPrior.
        """
        if not isinstance(prior,
                          (models.MvnModel,
                           models.BinomialLogitMvnPrior,
                           models.BinomialLogitSpikeSlabPrior)):
            raise Exception("""
            Unrecognized class of prior distribution passed to
            'LogitBandit.set_prior'.
            """)
        self._prior = prior

    @property
    def coefficient_draws(self):
        if not self._boom_bandit:
            return None
        else:
            return R.to_numpy(self._boom_bandit.coefficient_draws)

    @property
    def log_likelihood(self):
        """
        Return the log likelihood associated with each MCMC draw of model
        coefficients.  If the internal model has not yet been instantiated then
        return None.
        """
        if not self._boom_bandit:
            return None
        else:
            return R.to_numpy(self._boom_bandit.log_likelihood)

    def set_coefficient_draws(self, draws):
        """
        Populate the internal model with a set of coefficient draws.  This
        is mainly useful for deserializing a previously stored model.
        """
        self.boom().set_coefficient_draws(R.to_boom_matrix(draws))

    def set_log_likelihood(self, log_likelihood):
        """
        Populate the internal model with a set of log likelihood values
        associated with a previous MCMC run.
        """
        self.boom().set_log_likelihood(R.to_boom_vector(log_likelihood))

    def observe_past_data(self, successes, trials, features):
        if self._training_data:
            raise Exception("Cannot mix calls to 'observe_past_data' with "
                            "calls to 'observe_data'.")
        if not isinstance(features, pd.DataFrame):
            raise Exception("'features' argument must be a pandas data frame.")
        self._training_data = features
        self._training_data["successes"] = successes
        self._training_data["trials"] = trials

        if self._boom_model:
            self._boom_model.add_dataset(
                R.to_boom_vector(successes),
                R.to_boom_vector(trials),
                R.to_boom_matrix(self._encoder.encode_dataset(self._training_data)))
    
    def observe_data(self, arm: int, successes: int, trials: int, context=None):
        """
        Record observed outcomes for a given arm.

        Args:
          arm: The arm index (0-based).
          successes: Number of successes observed.
          trials: Number of trials observed.
          context: A single-row pandas DataFrame with context variables, or
            None if there are no context variables.
        """
        if context is not None and not isinstance(context, pd.DataFrame):
            raise TypeError("context must be a single-row pandas DataFrame or None.")

        self._training_data.append({
            "arm": int(arm),
            "successes": int(successes),
            "trials": int(trials),
            "context": context,
        })

    def update_posterior(self, ndraws: int):
        """
        Draw samples from the posterior distribution of the logistic
        regression coefficients.

        Args:
          ndraws: Number of posterior samples to draw.
        """
        self.boom().update_posterior(int(ndraws))

    @property
    def ndraws(self):
        """The number of posterior draws from the most recent update_posterior call."""
        return self.boom().ndraws

    def value(self, arm: int, context=None):
        """
        Return the predicted success probability for the given arm under the
        current model parameters.

        Args:
          arm: The arm index (0-based).
          context: A single-row pandas DataFrame, or None.
        """
        return self.boom().value(int(arm), _to_boom_context(context))

    def optimal_arm_probabilities(self, context=None):
        """
        Return the Thompson sampling probability that each arm is optimal,
        using the posterior draws from the most recent update_posterior call.

        Args:
          context: A single-row pandas DataFrame, or None.

        Returns:
          A numpy array of probabilities, one per arm, summing to 1.
        """
        return R.to_numpy(
            self.boom().optimal_arm_probabilities(_to_boom_context(context)))

    def thompson(self, context=None):
        """
        Return one draw of Thompson sampling for the bandit.  This does not
        update the posterior distribution.  It samples one set of model
        parameters from the set of posterior draws, calls
        'optimal_arm_probabilities' assuming that draw is the true set of
        parameters, and returns the values of the chosen arm.

        Args:
          context: A single-row pandas DataFrame object.  It can also be None if
            the bandit contains no contextual variables in its encoder.

        Returns:
          A list of strings giving the values of the action variables for the
          chosen arm.
        """
        return self.boom().thompson(_to_boom_context(context))
        
    @property
    def last_thompson_row(self):
        return self.boom().last_thompson_row

    @property
    def last_thompson_arm(self):
        return self.boom().last_thompson_arm

    def value_remaining_distribution(self, context=None):
        """
        Return the distribution of value remaining given context.

        Args:
          context: A single-row pandas DataFrame, or None.

        Returns:
          A numpy array of length ndraws, where each element is the difference
          between the best arm's predicted probability and arm 0's probability
          in that posterior draw.
        """
        return R.to_numpy(
            self.boom().value_remaining_distribution(_to_boom_context(context)))

    def arm_predictors(self, context=None):
        """
        Return the predictor matrix for all arms given the context.

        Args:
          context: A single-row pandas DataFrame, or None.

        Returns:
          A numpy array with one row per arm and one column per predictor.
        """
        return R.to_numpy(
            self.boom().arm_predictors(_to_boom_context(context)))

    def boom(self):
        """
        Instantiate any missing boom objects and return the underlying
        boom.LogitBandit.
        """
        if not self._boom_bandit:
            self._boom_model = self._define_model()
            self._boom_sampler = self._define_sampler(self._boom_model)
            self._boom_model.set_method(self._boom_sampler)
            if self._value_function:
                self._boom_bandit = boom.LogitBanditExternalValue(
                    self._boom_model,
                    self._encoder.boom(),
                    self._value_function)
            else:
                self._boom_bandit = boom.LogitBandit(
                    self._boom_model,
                    self._encoder.boom())
        return self._boom_bandit

    def _define_model(self):
        xdim = self._encoder.dim
        model = boom.BinomialLogitModel(xdim)
        if isinstance(self._training_data, list) and self._training_data:
            n = len(self._training_data)
            predictor_matrix = np.zeros((n, xdim))
            successes = np.zeros(n)
            trials = np.zeros(n)
            for i, obs in enumerate(self._training_data):
                predictor_matrix[i, :] = self._encoder.encode_row(
                    obs["arm"], obs["context"])
                successes[i] = obs["successes"]
                trials[i] = obs["trials"]
            model.add_dataset(
                R.to_boom_vector(successes),
                R.to_boom_vector(trials),
                R.to_boom_matrix(predictor_matrix))
        elif isinstance(self._training_data, pd.DataFrame):
            predictors = self._encoder.encode_dataset(self._training_data)
            successes = self._training_data["successes"].astype(float)
            trials = self._training_data["trials"].astype(float)
            model.add_dataset(R.to_boom_vector(successes),
                              R.to_boom_vector(trials),
                              R.to_boom_matrix(predictors))
        return model

    def _define_sampler(self, model):
        """

        Args:
          model:  A boom.BinomialLogitModel object to use as a sampling target.

        Returns:
          A boom posterior sampler object that can be assigned to the model.
        """
        if self._prior is None:
            # default value: This is not a great default.
            dim = self._encoder.dim
            mu = np.zeros(dim)
            Sigma = np.diag(np.ones(dim))
            self._prior = models.MvnModel(mu, Sigma)
            return boom.BinomialLogitAuxmixSampler(model, self._prior.boom())
        
        if isinstance(self._prior, models.MvnModel):
            return boom.BinomialLogitAuxmixSampler(model, self._prior.boom())

        elif isinstance(self._prior, models.BinomialLogitMvnPrior):
            return self._prior.create_sampler(model)

        elif isinstance(self._prior, models.BinomialLogitSpikeSlabPrior):
            return self._prior.create_sampler(model)

        else:
            raise Exception(
                """
                Unrecognized self._prior model family in call to
                _define_sampler.
                """)
                        

    def __getstate__(self):
        return {
            "arm_map": self._arm_map,
            "encoder": self._encoder,
            "value_function": self._value_function,
            "training_data": self._training_data,
            "coefficient_draws": self.coefficient_draws,
            "log_likelihood": self.log_likelihood,
            "prior": self._prior,
        }

    def __setstate__(self, payload):
        self._arm_map = payload["arm_map"]
        self._encoder = payload["encoder"]
        self._value_function = payload["value_function"]
        self._training_data = payload["training_data"]
        self._boom_model = None
        self._boom_sampler = None
        self._boom_bandit = None
        # Restore prior before boom() is first called so _define_sampler picks it up.
        self._prior = payload.get("prior", None)
        coef_draws = payload["coefficient_draws"]
        log_lik = payload["log_likelihood"]
        if coef_draws is not None:
            self.set_coefficient_draws(coef_draws)
        if log_lik is not None:
            self.set_log_likelihood(log_lik)


def _to_boom_context(context):
    if context is None:
        return boom.MixedMultivariateData()
    return R.to_boom_mixed_data(context)


class ValueFunctionJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        value_function_type = obj.__class__.__name__
        global VALUE_FUNCTION_JSON_ENCODER_REGISTRY
        if value_function_type not in VALUE_FUNCTION_JSON_ENCODER_REGISTRY:
            raise Exception(
                f"{value_function_type} was not found in the value function "
                "JSON encoder registry.  Please register the type, along with "
                "its JSON encoder and decoder, using "
                "register_value_function_json_encoder.")
            
        enc = VALUE_FUNCTION_JSON_ENCODER_REGISTRY[value_function_type]()
        payload = {
            "type": value_function_type,
            "function": enc.default(obj)
        }
        return payload


class ValueFunctionJsonDecoder(json.JSONDecoder):
    def decode(self, json_str):
        payload = json.loads(json_str)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        value_type = payload["type"]
        global VALUE_FUNCTION_JSON_DECODER_REGISTRY
        if value_type not in VALUE_FUNCTION_JSON_DECODER_REGISTRY:
            raise Exception(
                f"{value_type} was not found in the value function "
                "JSON decoder registry.  Please register the type, along with "
                "its JSON encoder and decoder, using "
                "register_value_function_json_encoder.")

        decoder = VALUE_FUNCTION_JSON_DECODER_REGISTRY[value_type]()
        return decoder.decode_from_dict(payload["function"])
    

class BinomialLogitPriorJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, models.MvnModel):
            return {
                "type": "MvnModel",
                "mu": obj._mu.tolist(),
                "Sigma": obj._Sigma.tolist(),
            }
        elif isinstance(obj, models.BinomialLogitMvnPrior):
            return {
                "type": "BinomialLogitMvnPrior",
                "mu": obj._mu.tolist() if obj._mu is not None else None,
                "Sigma": obj._Sigma.tolist() if obj._Sigma is not None else None,
                "variance_scale": obj._variance_scale,
                "clt_threshold": obj._clt_threshold,
            }
        elif isinstance(obj, models.BinomialLogitSpikeSlabPrior):
            return {
                "type": "BinomialLogitSpikeSlabPrior",
                "mu": obj._mu.tolist() if obj._mu is not None else None,
                "Sigma": obj._Sigma.tolist() if obj._Sigma is not None else None,
                "variance_scale": obj._variance_scale,
                "expected_model_size": obj._expected_model_size,
                "clt_threshold": obj._clt_threshold,
            }
        else:
            raise ValueError(f"Unsupported prior type: {type(obj).__name__}")


class BinomialLogitPriorJsonDecoder(json.JSONDecoder):
    def decode(self, json_str):
        return self.decode_from_dict(json.loads(json_str))

    def decode_from_dict(self, payload):
        type_name = payload["type"]
        if type_name == "MvnModel":
            return models.MvnModel(
                np.array(payload["mu"]),
                np.array(payload["Sigma"]),
            )
        elif type_name == "BinomialLogitMvnPrior":
            return models.BinomialLogitMvnPrior(
                mu=np.array(payload["mu"]) if payload["mu"] is not None else None,
                Sigma=(np.array(payload["Sigma"])
                       if payload["Sigma"] is not None else None),
                variance_scale=payload["variance_scale"],
                clt_threshold=payload["clt_threshold"],
            )
        elif type_name == "BinomialLogitSpikeSlabPrior":
            return models.BinomialLogitSpikeSlabPrior(
                mu=np.array(payload["mu"]) if payload["mu"] is not None else None,
                Sigma=(np.array(payload["Sigma"])
                       if payload["Sigma"] is not None else None),
                variance_scale=payload["variance_scale"],
                expected_model_size=payload["expected_model_size"],
                clt_threshold=payload["clt_threshold"],
            )
        else:
            raise ValueError(f"Unsupported prior type: {type_name}")


class LogitBanditJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        payload = {}

        arm_map_encoder = ArmMapJsonEncoder()
        payload["arm_map"] = arm_map_encoder.default(obj._arm_map)

        encoder_encoder = LinearBanditEncoderJSONEncoder()
        payload["encoder"] = encoder_encoder.default(obj._encoder)

        payload["log_likelihood"] = obj.log_likelihood

        if obj._prior is not None:
            payload["prior"] = BinomialLogitPriorJsonEncoder().default(obj._prior)

        if (obj._value_function is not None):
            value_encoder = ValueFunctionJsonEncoder()
            payload["value_function"] = value_encoder.default(
                obj._value_function)

        return payload


class LogitBanditJsonDecoder(json.JSONDecoder):
    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        arm_map_decoder = ArmMapJsonDecoder()
        arm_map = arm_map_decoder.decode_from_dict(payload["arm_map"])

        encoder_decoder = LinearBanditEncoderJSONDecoder()
        bandit_encoder = encoder_decoder.decode_from_dict(payload["encoder"])

        if "value_function" in payload:
            value_decoder = ValueFunctionJsonDecoder()
            value_function = value_decoder.decode_from_dict(
                payload["value_function"])
        else:
            value_function = None

        ans = LogitBandit(arm_map, bandit_encoder, value_function)

        # Restore prior before set_log_likelihood so boom()/_define_sampler picks it up.
        if "prior" in payload:
            ans._prior = BinomialLogitPriorJsonDecoder().decode_from_dict(payload["prior"])

        if payload["log_likelihood"] is not None:
            ans.set_log_likelihood(payload["log_likelihood"])

        return ans
