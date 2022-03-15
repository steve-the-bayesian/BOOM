# import matplotlib.pyplot as plt
# import seaborn as sns
import BayesBoom.boom as boom
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import pickle
import time


class MixedDataImputer:
    """
    Imputes missing data in a pd.DataFrame containing a mix of numeric and
    categorical data.

    Example use:
    """

    def __init__(self):
        """
        Create an empty MixedDataImputer.
        """
        self._model = None
        self._numeric_colnames = None
        self._categorical_colnames = None

        # A dict keyed by variable names of numeric columns.  The values are
        # lists of numeric values to be treated as atoms for that variable.
        self._atoms = {}

        self._levels = {}

        # A dict keyed by variable names of numeric columns.  The values are
        # 1-d numpy arrays containing prior counts for the atom values for that
        # variable.  If there are 'k' atoms then the vector must be of length
        # 'k+1', with the final count corresponding to the implicit continuous
        # atom.
        self._atom_prior = {}

        # A dict keyed by variable names of numeric columns.  The values are
        # 1-d numpy arrays containing prior counts for the level values for
        # that variable.  If there are 'k' levels then the vector must be of
        # length 'k'.
        self._level_prior = {}

        self._dataset_encoder = None
        self.coefficient_draws = np.zeros((0, 0))
        self.residual_variance_draws = np.zeros(0)
        self.atom_probs = None

    def save(self, filename):
        """
        Save the object to a pickle file.

        Args:
          filename: The name of the file in which to save the state of this
            object.

        Effect:
          The state of this object is written to 'filename'.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load a previously fit model from a pickle file.

        Args:
          filename:  The name of the file from which to load state.

        Effects:
          The state of this object is overwritten by the saved state in
          'filename'.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    @property
    def nclusters(self):
        """
        The number of clusters in the mixture model.
        """
        if self._model is None:
            return 0
        else:
            return self._model.nclusters

    @property
    def xdim(self):
        """
        The dimension of the dummy variable expansion of the categorical
        variables.
        """
        if self._model is None:
            return 0
        else:
            return self._model.xdim

    @property
    def ydim(self):
        """
        The number of numeric variables.
        """
        if self._numeric_colnames is None:
            return 0
        return len(self._numeric_colnames)

    @property
    def niter(self):
        """
        The number of iterations in the model.
        """
        return self.coefficient_draws.shape[0]

    def find_atoms(self, data: pd.DataFrame):
        """
        Fill a dict associating each numeric column name in data with a flat
        numpy array containing the atoms in that variable.

        Args:
          data: The data frame in which to search for atoms.

        Effect:
          self._numeric_colnames:
            If already populated then it is left unchanged.  Otherwise, it is
            filled with a list of strings giving the names of the numeric
            columns in 'data'.
          self._atoms: Filled with a dict containing the atoms in each of the
            numeric columns.  The dict is keyed by column names of the numeric
            columns.  A numeric column with no atoms is marked by an empty
            list.
        """
        if self._numeric_colnames is None:
            self._discover_column_types(data)

        for vname in self._numeric_colnames:
            variable = data[vname]
            counts = variable.value_counts().sort_values(ascending=False)
            number_observed = counts.sum()
            atom_indicator = (counts > .05 * number_observed)
            self._atoms[vname] = counts[atom_indicator].index.tolist()
            if len(self._atoms[vname]) > 3:
                self._atoms[vname] = self._atoms[vname][:3]
        print(f"Atoms: {pd.Series(self._atoms)}")

    def find_levels(self, data: pd.DataFrame):
        """
        Fill in a dict associating each categorical column name in a data frame
        with the levels of the categorical variable contained in the column.

        Effect:
          self._categorical_colnames: If already populated then it is left
            unchanged.  Otherwise it is filled with a list of strings giving
            the names fo the categorical variables in 'data'.
          self._levels: Filled with a dict containing the levels of each
            categorical column in 'data'.  The dict is keyed by variable names,
            and its elements are lists of strings.
        """
        if self._categorical_colnames is None:
            self._discover_column_types(data)

        for vname in self._categorical_colnames:
            variable = data[vname]
            levels = variable.unique()
            self._levels[vname] = levels.tolist()

    def set_level_prior(self, variable_name: str, prior_counts: np.ndarray):
        """
        Record the prior distribution on the distribution of the level
        probabilities for the specified variable.  Each cluster will use these
        prior counts as the prior distribution for this variable.

        Args:
          variable_name: A string matching the name of one of the categorical
            variables.
          prior_counts: A 1-d numpy array.  All entries must be positive, and
            the number of entries must match the number of levels for the
            specified variable.
        """
        if not np.all(prior_counts > 0):
            raise Exception("All prior counts must be positive.")
        if variable_name not in self._categorical_colnames:
            raise Exception(f"Could not find {variable_name} in categorical "
                            f"column names: {self._categorical_colnames}")

    def set_atom_prior(self, variable_name: str, prior_counts: np.ndarray):
        """
        Record the prior distribution on the distribution of atom values for the
        specified variable.  Each cluster entry uses this as its prior
        distribution.

        Args:
          variable_name: A string matching the name of one of the numeric
            variables.
          prior_counts: A 1-d numpy array.  If there are 'k' atoms then the
            array needs k+1 entries, with the last entry corresponding to the
            continuous atom.  A zero or negative entry indicates a strong prior
            assumption that the corresponding atom is never the true value.

        Effect:
          An entry is registered in self._atom_prior.  These entries are
          promoted to posterior samplers when the model is trained.
        """
        if np.sum(prior_counts > 0) == 0:
            raise Exception("At least one prior count must be positive.")
        if np.any(prior_counts < 0):
            raise Exception("Prior counts cannot be negative.")
        if variable_name not in self._atoms.keys():
            msg = "Attempt to set a prior for unrecognized variable "
            msg += f"{variable_name}."
            raise Exception
        if len(prior_counts) != len(self._atoms[variable_name]) + 1:
            raise Exception("The dimension of 'prior_counts' must exceed the"
                            " number of atoms by 1.")
        self._atom_prior[variable_name] = prior_counts.ravel()

    def train_model(self,
                    data: pd.DataFrame,
                    nclusters: int,
                    niter: int = 1000,
                    checkpoint_filename="",
                    nthreads=1):
        """
        Train the imputation model by taking 'niter' draws from the posterior
        distribution given a set of training data.

        Args:
          data: The data on which to train the model.  Any non-numeric
            variables are treated as categorical.  Missing values are expected
            to be coded as np.NaN.
          nclusters: The number of clusters to use for the joint
            distribution of the categorical data.
          niter:  The number of MCMC iterations to use during training.
          checkpoint_filename: The name of a file to use when recording the
            model's state.  The model will checkpoint its history every so
            often during training.
          nthreads: The number of threads to use during training.  This is
            experimental.

        Effects:
          self._coefficient_draws: Created and populated by MCMC draws of the
            linear regression coefficients.

        """
        self._start_time = time.time()
        if self._numeric_colnames is None or self._categorical_colnames is None:
            self._discover_column_types(data)

        # Find the set of atoms for each numeric variable.
        if self._atoms is None:
            self.find_atoms(data.loc[:, self._numeric_colnames])
        if len(self._atoms) != len(self._numeric_colnames):
            raise Exception(
                "One atom vector is needed for each numeric variable.")

        # Find the set of levels for each categorical variable.
        if self._levels is not None:
            self.find_levels(data.loc[:, self._categorical_colnames])
        if len(self._levels) != len(self._categorical_colnames):
            raise Exception(
                "Each categorical column must have its own list of levels.")

        # Convert the atoms to BOOM::Vector objects.
        atom_vector = []
        for vname in self._numeric_colnames:
            atoms = np.array(self._atoms[vname])
            atom_vector.append(boom.Vector(atoms.flatten().astype("float")))

        # The data table as a BOOM object.
        data_table = boom.to_data_table(data)

        self._model = boom.MixedDataImputer(
            nclusters, data_table, atom_vector, boom.GlobalRng.rng)

        # if nthreads > 1:
        #     self._model.setup_worker_pool(nthreads)

        print("setting prior")
        self._set_prior()

        print("Allocating space")
        self._allocate_space(niter)

        for i in range(niter):
            if i % 100 == 0:
                sep = "=-=-=-=-=-=-=-=-="
                print(f"{sep} {time.asctime()} Iteration {i} {sep}")
            self._model.sample_posterior()
            self._record_draws(i)
            if i % niter == 0 and checkpoint_filename != "":
                self.save(checkpoint_filename)
        self._end_time = time.time()

    def impute_rows(self, data, iterations):
        """
        Args:
          data:  Data frame containing the rows to impute.
          iterations:  An array-like collection of iteration numbers to use.

        Returns:
          A list of imputed data frames.

        Example:
        """
        imputed = []
        table = boom.to_data_table(data)
        for it in iterations:
            self._restore_parameters(it)
            imputed.append(
                boom.to_pandas(
                    self._model.impute_data_set(table, burn=20)))
        return imputed

    def _discover_column_types(self, data):
        """
        Populate self._numeric_colnames with a list containing the names of the
        numeric columns in 'data'.
        """
        self._numeric_colnames = []
        self._categorical_colnames = []

        for vname in data.columns:
            maybe_numeric = is_numeric_dtype(data.loc[:, vname])
            if maybe_numeric:
                counts = data.loc[:, vname].value_counts()
                if counts.shape[0] > 20:
                    self._numeric_colnames.append(vname)
            if self._numeric_colnames[-1] != vname:
                self._categorical_colnames.append(vname)

    def _format_imputation_data(self, data):
        dummies = self.encode(data)
        numerics = data.loc[:, self._numeric_colnames]
        formatted_data = []
        for i in range(data.shape[0]):
            y = boom.Vector(
                numerics.iloc[i, :].values.flatten().astype("float"))
            x = boom.Vector(dummies[i, :].flatten().astype("float"))
            formatted_data.append(boom.MvRegData(y, x))
        return formatted_data

    def _allocate_space(self, niter):
        """
        Allocate space to hold MCMC draws.

        Args:
          niter:  The desired number of draws.
        """
        self.coefficient_draws = np.empty((niter, self.xdim, self.ydim))
        self.residual_variance_draws = np.empty((niter, self.ydim, self.ydim))
        self.atom_probs = {}
        self.atom_error_probs = {}

        for vname, atom in self._atoms.items():
            k = len(atom)
            self.atom_probs[vname] = np.empty((niter, self.nclusters, k + 1))
            self.atom_error_probs[vname] = np.empty(
                (niter, self.nclusters, k + 1, k + 2))

    def _record_draws(self, i):
        """
        Record draws of the model parameters at iteration i.
        """
        self.coefficient_draws[i, :, :] = self._model.coefficients.to_numpy()
        self.residual_variance_draws[i, :, :] = (
            self._model.residual_variance.to_numpy()
        )
        for cluster in range(self.nclusters):
            for col in range(len(self._numeric_colnames)):
                name = self._numeric_colnames[col]
                self.atom_probs[name][i, cluster, :] = (
                    self._model.atom_probs(cluster, col).to_numpy()
                )

    def _restore_parameters(self, i):
        self._model.set_coefficients(boom.Matrix(
            self.coefficient_draws[i, :, :]))
        self._model.set_residual_variance(boom.Matrix(
            self.residual_variance_draws[i, :, :]))
        for cluster in range(self.nclusters):
            for col in range(len(self._numeric_colnames)):
                name = self._numeric_colnames[col]
                probs = self.atom_probs[name][i, cluster, :]
                self._model.set_atom_probs(cluster, col, boom.Vector(probs))

                error_probs = self.atom_error_probs[name][i, cluster, :, :]
                self._model.set_atom_error_probs(
                    cluster, col, boom.Matrix(error_probs))

    def _set_prior(self):
        """
        Set the prior distribution on the imputation model, and assign a
        PosteriorSampler.

        A nearly noninformative prior is chosen for the
        residual variance matrix and regression coefficients.  Each cell in in
        the mixture model is assigned an identical prior.
        """
        if self._model is None:
            raise Exception(
                "_set_prior was called before the model was created.")

        self._set_regression_prior()
        self._set_mixing_distribution_prior()
        for i, vname in enumerate(self._numeric_colnames):
            if vname not in self._atom_prior.keys():
                self._atom_prior[vname] = self._default_atom_prior(
                    self._atoms[vname])
            self._model.set_atom_prior(
                boom.Vector(self._atom_prior[vname].astype("float")),
                i)

        for i, vname in enumerate(self._categorical_colnames):
            if vname not in self._level_prior.keys():
                self._level_prior[vname] = self._default_level_prior(
                    self._levels[vname])
            self._model.set_level_prior(
                boom.Vector(self._level_prior[vname].astype("float")),
                i)

    def _set_mixing_distribution_prior(self):
        prior_counts = np.ones(self.nclusters)
        prior_counts /= len(prior_counts)
        self._model.set_mixing_weight_prior(boom.Vector(
            prior_counts.astype("float")))

    def _set_regression_prior(self):
        coefficient_prior_mean = np.zeros((self.xdim, self.ydim))
        coefficient_prior_mean[:, 0] = 0.0    # TODO replace 0.0 with ybar

        coefficient_weight = 1.0
        variance_weight = float(self.ydim + 1)
        residual_variance_guess = np.eye(self.ydim)

        self._model.set_regression_prior(
            boom.Matrix(coefficient_prior_mean.astype("float")),
            float(coefficient_weight),
            boom.SpdMatrix(residual_variance_guess.astype("float")),
            float(variance_weight))

    def _default_atom_prior(self, atoms):
        """
        Return a default prior distribution for the vector of atoms.
        """
        prior_counts = np.ones(len(atoms) + 1, dtype="float")
        prior_counts /= np.sum(prior_counts)
        return prior_counts

    def _default_level_prior(self, levels):
        """
        Return a default prior distribution for the vector of levels.
        """
        prior_counts = np.ones(len(levels), dtype="float")
        prior_counts /= np.sum(prior_counts)
        return prior_counts

    def __getstate__(self):
        """
        Return a dict that can be used to pickle a MixedDataImputer.
        """
        state = {
            "atoms": self._atoms,
            "numeric_colnames": self._numeric_colnames,
            "categorical_colnames": self._categorical_colnames,
            "atom_prior": self._atom_prior,
            "level_prior": self._level_prior,
            "coefficient_draws": self.coefficient_draws,
            "residual_variance_draws": self.residual_variance_draws,
            "atom_probs": self.atom_probs,
            "nclusters": self.nclusters,
            "empirical_distributions": self._model.empirical_distributions,
        }
        return state

    def __setstate__(self, state):
        """
        Retrieve a MixedDataImputer from a pickle.
        """
        self._atoms = state["atoms"]
        self._numeric_colnames = state["numeric_colnames"]
        self._categorical_colnames = state["categorical_colnames"]
        self._atom_prior = state["atom_prior"]
        self._dataset_encoder = state["dataset_encoder"]
        self.coefficient_draws = state["coefficient_draws"]
        self.residual_variance_draws = state["residual_variance_draws"]
        self.atom_probs = state["atom_probs"]
        self.empirical_distributions = state["empirical_distributions"]

        if (
                self._dataset_encoder is not None
                and state["nclusters"] is not None
        ):
            atom_vector = []
            for vname in self._numeric_colnames:
                atoms = np.array(self._atoms[vname])
                atom_vector.append(boom.Vector(
                    atoms.flatten().astype("float")))
            self._model = boom.MixedDataImputer(
                # TODO: this is hosed.
            )
            # state["nclusters"], atom_vector, xdim
            self._model.set_empirical_distributions(
                state["empirical_distributions"])
