import BayesBoom.boom as boom
import BayesBoom.R as R
import numpy as np
import pandas as pd
import json

class ExperimentStructure:
    """
    A collection of all factor names and factor level values defining the
    experiment.
    """
    def __init__(self):
        self._factor_names = []
        self._factor_levels = []
        self._baseline_levels = []
        self._boom_experiment_structure = None

    def add_factor(self,
                   factor_name: str,
                   factor_levels,
                   baseline_level: str = ""):
        """
        Add a factor to the experiment.

        Args:
          factor_name: The name of the experimental factor.  Think of this as
            the variable name in a data frame.
          factor_levels: The possible values the factor can assume.  A list of
            strings.
          baseline_level: The level of the factor to leave out when creating
            dummy variables.

        Effects:
          The internal structure is updated to reflect the additional factor and
          levels.
        """
        self._factor_names.append(str(factor_name))
        self._factor_levels.append([str(level) for level in factor_levels])
        self._baseline_levels.append(str(baseline_level))

        if self._boom_experiment_structure:
            self._boom_experiment_structure.add_factor(
                self._factor_names[-1],
                self._factor_levels[-1])

            
    def __getitem__(self, factor_name):
        pos = self._factor_names.index(factor_name)
        return self._factor_levels[pos]

            
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

    
# ===========================================================================    
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

    @property
    def structure(self):
        return self._experiment_structure

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


# ===========================================================================    
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

        if baseline_level == "":
            self._baseline_level = self.levels[-1]

    @property
    def dim(self):
        return self.boom().dim

    @property
    def encoded_variable_names(self):
        return [f"{self._variable_name}[{i + 1}]" for i in range(self.dim)]

    @property
    def levels(self):
        return self._arm_map.structure[self._variable_name].copy()

    def encode_dataset(self, data):
        if self._variable_name not in data.columns:
            raise Exception(
                f"{self._variable_name} is not a named variable in the "
                "data set.")
        x = data.loc[:, self._variable_name]
        levels = self._arm_map.structure[self._variable_name].copy()
        levels.remove(self._baseline_level)
        sample_size = x.shape[0]
        ans = np.zeros((sample_size, self.dim))
        for i in range(self.dim):
            ans[:, i] = (x == levels[i]).astype(float)
        baseline = x == self._baseline_level
        ans[baseline, :] = -1
        return ans

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

        
class ExperimentArmEncoderJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        arm_map_encoder = ArmMapJsonEncoder()
        payload = {
            "variable_name": str(obj._variable_name),
            "arm_map": arm_map_encoder.default(obj._arm_map),
            "baseline_level": str(obj._baseline_level)
        }
        return payload

    
class ExperimentArmEncoderJSONDecoder(json.JSONDecoder):
    
    def decode(self, json_string):
        payload = json.loads(json_string)
        return self.decode_from_dict(payload)
    
    def decode_from_dict(self, payload):
        arm_map_decoder = ArmMapJsonDecoder()
        arm_map = arm_map_decoder.decode_from_dict(payload["arm_map"])
        return ExperimentArmEncoder(
            payload["variable_name"],
            arm_map,
            payload["baseline_level"])

    
R.register_encoding_json_encoder(
    "ExperimentArmEncoder",
    ExperimentArmEncoderJSONEncoder,
    ExperimentArmEncoderJSONDecoder)
    

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
            boom_context = boom.MixedMultivariateData()
        else:
            boom_context = R.to_boom_mixed_data(context)
        return R.to_numpy(self._boom_encoder.encode_row(int(arm), boom_context))

    def encode_dataset(self, context):
        """
        """
        return self._dataset_encoder.encode_dataset(context)
            


class LinearBanditEncoderJSONEncoder(json.JSONEncoder):

    def default(self, obj):
        arm_map_encoder = ArmMapJsonEncoder()
        dataset_json_encoder = R.DatasetEncoderJsonEncoder()
        payload = {
            "dataset_encoder": dataset_json_encoder.default(
                obj._dataset_encoder),
            "arm_map": arm_map_encoder.default(obj._arm_map)
        }
        return payload

    
class LinearBanditEncoderJSONDecoder(json.JSONDecoder):
    def decode(self, json_string):
        return self.decode_from_dict(json.loads(json_string))
    
    def decode_from_dict(self, payload):
        arm_map_decoder = ArmMapJsonDecoder()
        dataset_encoder_json_decoder = R.DatasetEncoderJsonDecoder()
        
        arm_map = arm_map_decoder.decode_from_dict(payload["arm_map"])
        dataset_encoder = dataset_encoder_json_decoder.decode_from_dict(
            payload["dataset_encoder"])
        return LinearBanditEncoder(arm_map=arm_map,
                                   dataset_encoder=dataset_encoder)
        
        
