import BayesBoom.boom as boom
import BayesBoom.R as R
import numpy as np
import pandas as pd
import json


class ExperimentStructure:
    def __init__(self):
        self._factor_names = []
        self._factor_levels = []
        self._baseline_levels = []
        self._boom_experiment_structure = None

    def add_factor(self, factor_name, factor_levels, baseline_level=""):
        self._factor_names.append(str(factor_name))
        self._factor_levels.append([str(level) for level in factor_levels])
        self._baseline_levels.append(str(baseline_level))

        if self._boom_experiment_structure:
            self._boom_experiment_structure.add_factor(
                self._factor_names[-1],
                self._factor_levels[-1])

    def boom(self):
        if not self._boom_experiment_structure:
            self._boom_experiment_structure = boom.ExperimentStructure()
            for i in range(len(self._factor_names)):
                self._boom_experiment_structure.add_factor(
                    self._factor_names[i],
                    self._factor_levels[i])
        return self._boom_experiment_structure


class ExperimentStructureJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if not isinstance(obj, ExperimentStructure):
            raise Exception("Expected an ExperimentStructure object")
        payload = {
            "factor_names": obj._factor_names,
            "factor_levels": obj._factor_levels
        }
        return json.loads(json.dumps(payload))


class ExperimentStructureJSONDecoder(json.JSONDecoder):
    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        xp = ExperimentStructure()
        num_factors = len(payload["factor_names"])
        for i in range(num_factors):
            factor_name = str(payload["factor_names"][i])
            factor_levels = [str(level) for level in
                             payload["factor_levels"][i]]
            xp.add_factor(factor_name, factor_levels)
        return xp


class ArmMap:
    def __init__(self, experiment_structure):
        self._experiment_structure = experiment_structure
        self._boom_arm_map = None

    @property
    def number_of_arms(self):
        return self.boom().number_of_arms

    @property
    def map(self):
        return R.to_numpy(self.boom().factor_level_matrix)

    def boom(self):
        if not self._boom_arm_map:
            self._boom_arm_map = boom.ArmMap(self._experiment_structure.boom())
        return self._boom_arm_map


class ArmMapJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if not isinstance(obj, ArmMap):
            raise Exception("Json Encoder expected an ArmMap object")
        xp_enc = ExperimentStructureJSONEncoder()
        payload = {
            "experiment_structure": xp_enc.default(obj._experiment_structure)
        }
        return payload


class ArmMapJsonDecoder(json.JSONDecoder):
    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)

    def decode_from_dict(self, payload):
        xp_dec = ExperimentStructureJSONDecoder()
        xp = xp_dec.decode_from_dict(payload["experiment_structure"])
        return ArmMap(xp)


class ExperimentArmEncoder(R.Encoder):
    """
    Encodes one experimental factor as effects-coded predictors for use in a
    LinearBanditEncoder.  Arm assignment is handled by the parent
    LinearBanditEncoder; this encoder supplies the factor-level columns.
    """

    def __init__(self,
                 variable_name: str,
                 arm_map,
                 baseline_level: str = ""):
        self._variable_name = variable_name
        self._arm_map = arm_map
        self._baseline_level = baseline_level
        self._boom_encoder = None

    @property
    def dim(self):
        return self.boom().dim

    @property
    def encoded_variable_names(self):
        return [f"{self._variable_name}[{i + 1}]" for i in range(self.dim)]

    def encode_dataset(self, data):
        raise NotImplementedError(
            "Use LinearBanditEncoder.encode_row to encode (arm, context) pairs.")

    @property
    def required_variables(self):
        return [self._variable_name]

    def encodes(self, vname):
        return vname == self._variable_name

    def extract_main_effects(self):
        return {self._variable_name: self}

    def _create_boom_encoder(self):
        self._boom_encoder = boom.ExperimentArmEncoder(
            self._variable_name,
            self._arm_map.boom(),
            self._baseline_level)


class LinearBanditEncoder:
    """
    Converts (arm, context) pairs into predictor vectors for a generalized
    linear model.  The encoder combines ExperimentArmEncoder objects for the
    experimental factors with any additional context encoders.
    """

    def __init__(self, arm_map, dataset_encoder):
        self._arm_map = arm_map
        self._dataset_encoder = dataset_encoder
        self._boom_encoder = boom.LinearBanditEncoder(
            self._arm_map.boom(),
            self._dataset_encoder.boom())

    @property
    def number_of_arms(self):
        return self._arm_map.number_of_arms

    @property
    def dim(self):
        """The number of columns in the predictor vector produced by encode_row."""
        return self._dataset_encoder.dim

    def boom(self):
        return self._boom_encoder

    def encode_row(self, arm: int, context):
        """
        Encode an (arm, context) pair into a predictor vector.

        Args:
          arm: The arm index (0-based).
          context: A single-row pandas DataFrame with context variables, or
            None if there are no context variables.

        Returns:
          A 1-D numpy array of length self.dim.
        """
        if context is None or (isinstance(context, pd.DataFrame)
                                and context.shape[1] == 0):
            ctx = boom.MixedMultivariateData()
        else:
            ctx = R.to_boom_mixed_data(context)
        return R.to_numpy(self._boom_encoder.encode_row(int(arm), ctx))
