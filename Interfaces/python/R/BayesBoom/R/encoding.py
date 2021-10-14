import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# ===========================================================================
class Encoder(ABC):
    """
    An encoder converts a data frame (which might contain mixed type data:
    numeric, categorical, datetime) into a numeric matrix of predictors.

    The most common high-level encoder is a DatasetEncoder, which does the
    conversion mentioned above in a single call: encoder.encode_dataset(data).
    A DatasetEncoder is composed of other smaller encoders that do things like
    expand variables into spline basis expansions, dummy variables, and
    interactions.
    """

    @property
    @abstractmethod
    def dim(self):
        """
        The dimension of the basis element output by this encoder.
        """

    @property
    @abstractmethod
    def encoded_variable_names(self):
        """
        The column names of the predictor matrix produced by 'encode_dataset..
        """

    @abstractmethod
    def encode_dataset(self, data):
        """
        Args:
          data:  A pd.DataFrame or equivalent containing mixed-type data.

        Returns:
          predictors:  A np.ndarray containing the encoded numeric predictors.
        """

    @property
    @abstractmethod
    def required_variables(self):
        """
        A list of the variable names needed in order to call encode_dataset.
        """

    @abstractmethod
    def encodes(self, vname):
        """
        Returns True if the encoder encodes a variable named 'vname' and False
        otherwise.
        """


# ===========================================================================
class MainEffectEncoder:
    """
    A "Main Effect" is a variable in a model that is a function of a single
    column in the input data frame.
    """

    def __init__(self, variable_name: str):
        self._vname = variable_name

    @abstractmethod
    def encode(self, x):
        """
        Args:
          x: a pd.Series or equivalent.  Depending on the type of the concrete
            encoder, x might be numeric, categorical, datetime, or other
            non-numeric data type.

        Returns:
          encoded: A np.ndarray.  Each row matches the corresponding entry in
            'x'.  The number of columns is self.dim.
        """

    def encode_dataset(self, data):
        return self.encode(data.loc[:, self.variable_name])

    @property
    def variable_name(self):
        return self._vname

    @property
    def required_variables(self):
        return [self.variable_name]

    def encodes(self, vname):
        return vname == self.variable_name

    def __repr__(self):
        return str(self.__class__.__name__) + " for " + self.variable_name


# ===========================================================================
class EffectEncoder(MainEffectEncoder):
    """
    Convert categorical data into a matrix of dummy variables.  An
    EffectEncoder is similar to a "1-hot" encoder for all levels other than the
    baseline level.

    Each column of the output matrix corresponds to one non-baseline level of a
    categorical variable.
    - If the categorical is the "special" level that column gets the value 1.
    - If the categorical variable is neither the special level nor the baseline
      level then that column gets the value 0.
    - If the categorical variable is the baseline level then the column gets
      the value -1.

    The logic behind this encoding scheme is the effect of the baseline level
    is the negative sum of the effect of all other levels.  In other words, the
    set of effects for the categorical variable is constrained to sum to zero.

    Example:

      Suppose a variable Color is either Red, Blue, or Green, with Green being
      the baseline.  If the first three instances of Color happened to be Red,
      Blue, and Green in that order, the first three elements of the encoded
      output will be:

         Color.Red  Color.Blue
         1          0            # Red
         0          1            # Blue
        -1         -1            # Green

      Now suppose that the coefficient of "Color.Red" has the value 1.2, and
      the coefficient of Color.Blue has the value 2.3.  Multiply these
      coefficient by the rows of the encoded matrix to get

      The effect of "Red" = 1.2
      The effect of "Blue" = 2.3
      The effect of "Green" = -1.2 - 2.3.

      Obviously, the sum of these three effects is zero.

    Effect encoding has the appealing property that the an as-yet unseen level
    of the categorical variable.
    """

    def __init__(self, variable_name, levels, baseline_level=None):
        super().__init__(variable_name)
        self._levels = self._unique_levels(levels)
        assert isinstance(self._levels, list)

        if baseline_level is None:
            self._baseline = levels[-1]
        else:
            self._baseline = baseline_level
        if self._baseline in self._levels:
            self._levels.remove(self._baseline)

    @property
    def dim(self):
        return len(self._levels)

    def encode(self, factor):
        if not isinstance(factor, (np.ndarray, pd.Series)):
            factor = np.array(factor)
        nobs = len(factor)
        dim = self.dim
        ans = np.zeros((nobs, dim), dtype=float)
        for i in range(dim):
            ans[:, i] = (factor == self._levels[i]).astype(float)
        baseline = factor == self._baseline
        ans[baseline, :] = -1
        return ans

    @property
    def encoded_variable_names(self):
        return [self.variable_name + "." + str(x) for x in self._levels]

    @staticmethod
    def _unique_levels(levels):
        ans = []
        used = set()
        for level in levels:
            if level not in used:
                ans.append(level)
                used.add(level)
        return ans


# ===========================================================================
class IdentityEncoder(MainEffectEncoder):
    """
    An IdentityEncoder encodes a single numeric column into and n x 1 matrix.
    """

    def __init__(self, variable_name):
        super().__init__(variable_name)

    @property
    def dim(self):
        return 1

    def encode(self, x):
        return np.array(x).reshape(-1, 1)

    @property
    def encoded_variable_names(self):
        return self.variable_name


# ===========================================================================
class InteractionEncoder(Encoder):
    """
    Creates an interaction between two lower dimensional encoders.  Multi-way
    interactions can be implemented in terms of lower-order interactions.
    """

    def __init__(self, encoder1: Encoder, encoder2: Encoder):
        self._encoder1 = encoder1
        self._encoder2 = encoder2
        self._dim = self._encoder1.dim * self._encoder2.dim

    @property
    def dim(self):
        return self._encoder1.dim * self._encoder2.dim

    def encoded_variable_names(self):
        names1 = self._encoder1.encoded_variable_names
        names2 = self._encoder2.encoded_variable_names
        ans = []
        for name1 in names1:
            for name2 in names2:
                ans.append(name1 + ":" + name2)
        return ans

    def encode(self, factor1, factor2):
        nobs = len(factor1)
        if len(factor2) != nobs:
            raise Exception("Both factors in an interaction must have the "
                            "same number of observations.")

        enc1 = self._encoder1.encode(factor1)
        enc2 = self._encoder2.encode(factor2)
        return self.create_interaction(enc1, enc2)

    def create_interaction(self, basis1, basis2):
        """
        Returns a matrix that contains all column-wise products between the
        elements of basis1 and basis2, which are matrices with the same number
        of rows.
        """
        dim = basis1.shape[1] * basis2.shape[1]
        nobs = basis1.shape[0]
        if basis2.shape[0] != nobs:
            raise Exception("Basis matrices must have the same number of rows")
        ans = np.empty((nobs, dim))
        counter = 0
        for col1 in range(basis1.shape[1]):
            for col2 in range(basis2.shape[1]):
                ans[:, counter] = basis1[:, col1] * basis2[:, col2]
                counter += 1
        return ans

    def encode_dataset(self, data):
        return self.create_interaction(
            self._encoder1.encode_dataset(data),
            self._encoder2.encode_dataset(data))

    @property
    def required_variables(self):
        return list(set(self._encoder1.required_variables
                        + self._encoder2.required_variables))

    def encodes(self, vname):
        return self._encoder1.encodes(vname) or self._encoder2.encodes(vname)


# ===========================================================================
class DatasetEncoder(Encoder):

    def __init__(self, encoders, force_intercept: bool = True):
        self._encoders = encoders
        self._force_intercept = force_intercept

        self._initialize()

    @property
    def dim(self):
        return self._dim

    @property
    def encoded_variable_names(self):
        ans = []
        if self._force_intercept:
            ans.append("(Intercept)")
        for enc in self._encoders:
            ans += enc.encoded_variable_names
        return ans

    def encode_dataset(self, data: pd.DataFrame):
        """
        Encode the data set.

        Args:
          data: The data to be encoded.

        Returns:
          The matrix of encoded data (as an np.ndarray)
        """
        if data.shape[0] == 0:
            return np.empty((0, self.dim))
        encoded_variables = []
        if self._force_intercept:
            encoded_variables.append(np.ones((data.shape[0], 1)))
        for i in range(len(self._encoders)):
            encoded_variables.append(self._encoders[i].encode_dataset(data))
        if not encoded_variables:
            return np.array([])
        else:
            return np.concatenate(encoded_variables, axis=1)

    def _initialize(self):
        self._dim = self._force_intercept + np.sum(
            [x.dim for x in self._encoders])
        self._factor_map = self._create_variable_map()

    @property
    def required_variables(self):
        required = []
        for enc in self._encoders:
            required += enc.required_variables
        return list(set(required))

    def encodes(self, vname):
        for enc in self._encoders:
            if enc.encodes(vname):
                return True
        return False

    def _create_variable_map(self):
        vnames = self.required_variables
        affected_feature_indices = {}
        main_effect_indices = {}
        for name in vnames:
            affected, main = self._affected_features(name)
            affected_feature_indices[name] = affected
            main_effect_indices[name] = main
        return VariableMap(affected_feature_indices, main_effect_indices)

    def _affected_features(self, vname: str):
        """
        The set of output columns affected by the named variable.

        Args:
          vname:  The name of an input variable.

        Returns:
        - affected_indices: A list of column numbers affected by the named
          variable.
        - main_effect_indices: A list of column numbers affected by just the
          main effect of the named variable.
        """
        start = int(self._force_intercept)
        affected_indices = []
        main_effect_indices = []
        for enc in self._encoders:
            if enc.encodes(vname):
                affected_indices += list(range(start, start + enc.dim))
                if isinstance(enc, MainEffectEncoder):
                    main_effect_indices = list(range(start, start + enc.dim))
            start += enc.dim
        return affected_indices, main_effect_indices


# ===========================================================================
class VariableMap:
    """
    A mapping between columns in an expanded ("feature") matrix and the set of
    variables in the data frame that produced them.
    """

    def __init__(self, affected_features, main_effect_indices):
        """
        Args:
          affected_features: A dictionary, keyed by variable name.
            variable_map[vname] is a list of column indices in the encoded
            predictor matrix that are affected by variable 'vname' (main
            effects and interactions).
          main_effect_indices: A dictionary, keyed by variable name.
            main_effect_indices[vname] is a list of integers corresponding
            coulumn indices in the encoded predictor matrix that are part of
            main effects associated with vname.
        """
        self._variable_names = list(affected_features.keys())
        self._affected_features = affected_features
        self._main_effects_map = main_effect_indices
        self._interaction_mask = None

    # ---------------------------------------------------------------------------
    @property
    def variable_names(self):
        return self._variable_names

    # ---------------------------------------------------------------------------
    def included_variables(self, included_feature_matrix):
        """
        From a matrix of bools denoting included features (variables in the
        expanded matrix), return the corresponding matrix of included variables
        (in the raw data matrix).

        Args:
          included_feature_matrix: A numpy matrix of bools.  Each row denotes a
          subset of included features.  Each column corresponds to a feature in
          the expanded matrix.

        Returns:
          A pandas data frame, with rows corresponding to rows of
          'included_feature_matrix'.  Columns correspond to variables in the
          original data set.  The data frame contains boolean values, with True
          indicating that the variable was present in at least one main effect
          or interaction.
        """
        ans = np.zeros((included_feature_matrix.shape[0],
                        len(self._variable_names)), dtype=bool)

        for i in range(len(self._variable_names)):
            variable_name = self._variable_names[i]
            variable_map = self._affected_features[variable_name]
            variable_indicators = included_feature_matrix[:, variable_map]
            ans[:, i] = np.any(variable_indicators, axis=1)

        return pd.DataFrame(ans, columns=self._variable_names)

    # ---------------------------------------------------------------------------
    def included_interactions(self, included_feature_matrix):
        """
        Returns a pandas DataFrame indicating which two-factor interactions are
        present in the data.

        Args:
          included_feature_matrix: A numpy boolean matrix.
        """
        if self._interaction_mask is None:
            self._create_interaction_mask()

        if not self._interaction_mask:
            return None

        values = {}
        for terms, index in self._interaction_mask.items():
            key = terms[0] + ":" + terms[1]
            values[key] = np.any(included_feature_matrix[:, index], axis=1)
        return pd.DataFrame(values)

    # ---------------------------------------------------------------------------
    def main_effect_indices(self, name: str = None):
        """
        Returns the indices in the encoded matrix of all the main effect columns
        corresponding to the named variable.

        Args:
          name: The name of a variable appearing in a main effect, or None.

        Returns:
          If name is not None then the return value is a list containing the
          feature matrix indices corresponding to the main effect for 'name.'
            Otherwise the return value contains the indices for all main
            effects.
        """
        if name is not None:
            return self._main_effects_map[name]
        else:
            ans = []
            for key in self._main_effects_map.keys():
                ans = ans + self._main_effects_map[key]
            return ans

    # ---------------------------------------------------------------------------
    def interaction_indices(self, name1: str, name2: str):
        """
        The feature indices for the interaction between two variables.

        Args:
          name1: The name of the first variable in the interaction.
          name2: The name of the second variable in the interaction.

        Returns:
          A sorted list of the column indices for the set of features
          describing the interaction.
        """
        name1_indices = set(self._affected_features[name1])
        name2_indices = set(self._affected_features[name2])
        ans = list(name1_indices & name2_indices)
        ans.sort()
        return ans

    # ---------------------------------------------------------------------------
    def _create_interaction_mask(self):
        """
        Binds self._interaction_mask to a dictionary, keyed by pairs of variable
        names in self._variable_names.  The first entry in the pair always
        comes before the second in self._variable_names.  The dictionary entry
        is a list
        """
        self._interaction_mask = {}
        for i in range(len(self._variable_names)):
            for j in range(i + 1, len(self._variable_names)):
                factor1, factor2 = [self._variable_names[k] for k in [i, j]]
                interaction_columns = set(
                    self._affected_features[factor1]).intersection(
                        set(self._variable_map[factor2])
                    )
                if interaction_columns:
                    elements = list(interaction_columns)
                    elements.sort()
                    self._interaction_mask[(factor1, factor2)] = elements
