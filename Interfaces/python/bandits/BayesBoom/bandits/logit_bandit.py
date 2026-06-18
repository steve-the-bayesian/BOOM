import BayesBoom.boom as boom
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

    @property
    def number_of_arms(self):
        return self._encoder.number_of_arms

    @property
    def coefficient_draws(self):
        if not self._boom_bandit:
            return None
        else:
            return R.to_numpy(self._boom_bandit.coefficient_draws)

    def set_coefficient_draws(self, draws):
        self.boom().set_coefficient_draws(R.to_boom_matrix(draws))

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
                R.to_boom_matrix(predictors))
    
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

        # Reset boom objects so they are rebuilt from the complete training data
        # on the next call to boom().  BinomialLogitAuxmixSampler does not
        # support incremental data addition after sampling has started.
        self._boom_bandit = None
        self._boom_model = None
        self._boom_sampler = None

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
        prior = boom.MvnModel(self._encoder.dim, 0.0, 1.0)
        return boom.BinomialLogitAuxmixSampler(model, prior)


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
    

class LogitBanditJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        payload = {}
        
        arm_map_encoder = ArmMapJsonEncoder()
        payload["arm_map"] = arm_map_encoder.default(obj._arm_map)
        
        encoder_encoder = LinearBanditEncoderJSONEncoder()
        payload["encoder"] = encoder_encoder.default(obj._encoder)

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
        return ans
