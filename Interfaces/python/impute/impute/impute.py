# import matplotlib.pyplot as plt
# import seaborn as sns
import BayesBoom.boom as boom
import numpy as np
import pandas as pd
import pickle
import time


class MissingDataImputer:
    """
    Imputes missing data in a pd.DataFrame containing a mix of numeric and
    categorical data.
    """

    def __init__(self):
        """
        Create an empty MissingDataImputer.
        """
        self._model = None
        self._numeric_colnames = None
        self._categorical_colnames = None

        # A dict keyed by variable names of numeric columns.  The values are
        # lists of numeric values to be treated as atoms for that variable.
        self._atoms = {}

        # A dict keyed by variable names of numeric columns.  The values are
        # 1-d numpy arrays containing prior counts for the atom values for that
        # variable.  If there are 'k' atoms then the vector must be of length
        # 'k+1', with the final count corresponding to the implicit continuous
        # atom.
        self._atom_prior = {}

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
    def niter(self):
        """
        The number of iterations in the model.
        """
        return self.coefficient_draws.shape[0]

    def find_atoms(self, data: pd.DataFrame):
        """
        Fill a dict associating each column name in data with a flat numpy
        array containing the atoms in that variable.

        Args:
          data: The data frame in which to search for atoms.

        Effect:
          self._atoms is filled with a dict containing the atoms in each of the
          numeric columns.  The dict is keyed by column names of the numeric
          columns.  A numeric column with no atoms is marked by an empty list.
        """
        if self._numeric_colnames is None:
            self._discover_numeric_colnames(data)

        for vname in self._numeric_colnames:
            variable = data[vname]
            counts = variable.value_counts().sort_values(ascending=False)
            number_observed = counts.sum()
            atom_indicator = (counts > .05 * number_observed)
            self._atoms[vname] = counts[atom_indicator].index.tolist()
            if len(self._atoms[vname]) > 3:
                self._atoms[vname] = self._atoms[vname][:3]
        print(f"Atoms: {pd.Series(self._atoms)}")

    def set_categorical_encoders(self, data):
        if self._numeric_colnames is None:
            self._discover_numeric_colnames(data)

        encoders = []

        for vname in data.columns:
            if vname not in self._numeric_colnames:
                counts = data[vname].value_counts().sort_values(
                    ascending=False)
                if counts.shape[0] > 20:
                    counts = counts[:20]
                levels = counts.index.tolist()
                encoders.append(EffectsEncoder(
                    vname, levels=levels, baseline_level=levels[0]))

        self._encoder = DatasetEncoder(encoders, intercept=True)

    def encode(self, data):
        """
        Return a numpy array created by dummy-variable encoding the
        provided set of data.
        """
        if self._encoder is None:
            self.set_categorical_encoders(data)

        return self._encoder.encode_dataset(data)

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
        """
        self._atom_prior[variable_name] = prior_counts.ravel()
        if np.sum(prior_counts > 0) == 0:
            raise Exception("At least one prior count must be positive.")

    def train_model(self,
                    data: pd.DataFrame,
                    num_clusters: int,
                    niter: int = 1000,
                    checkpoint_filename="",
                    nthreads=1):
        """
        Train the imputation model by taking 'niter' draws from the posterior
        distribution given a set of training data.

        Args:
          data:
        """
        # Import is done here so that BayesBoom code will not be imported into
        # the tensorflow lambda.
        self._start_time = time.time()
        if self._numeric_colnames is None:
            print("discovering numeric columns")
            self.discover_numeric_colnames(data)

        self._cat_cols = [
            col for col in data.columns if col not in self._numeric_colnames
        ]

        if self._atoms is None:
            print("finding atoms")
            self.find_atoms(data.loc[:, self._numeric_colnames])

        if len(self._atoms) != len(self._numeric_colnames):
            raise Exception(
                "One atom vector is needed for each numeric variable.")

        print("extracting numerics")
        numerics = data.loc[:, self._numeric_colnames]
        ydim = numerics.shape[1]

        print("encoding dummy variables")
        dummies = self.encode(data.loc[:, self._cat_cols])
        xdim = dummies.shape[1]

        print("Converting atoms to vector")
        atom_vector = []
        for vname in self._numeric_colnames:
            atoms = np.array(self._atoms[vname])
            atom_vector.append(boom.Vector(atoms.flatten().astype("double")))

        print("initializing BOOM model")
        self._model = boom.MixedDataImputer(
            num_clusters, data_table, atom_vector, boom.GlobalRng.rng)

        if nthreads > 1:
            self._model.setup_worker_pool(nthreads)

        print("adding data")
        for i in range(data.shape[0]):
            y = boom.Vector(
                numerics.iloc[i, :].values.flatten().astype("float"))
            x = boom.Vector(dummies[i, :].flatten().astype("float"))
            self._model.add_data(boom.MvRegData(y, x))

        print("setting prior")
        self._set_prior(xdim, ydim)

        self._allocate_space(niter, xdim, ydim)

        for i in range(niter):
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

        """
        formatted_data = self._format_imputation_data(data)
        imputed = []

        for it in iterations:
            self._restore_parameters(it)
            imputed_matrix = self._model.impute_data_set(formatted_data)
            imputed.append(pd.DataFrame(imputed_matrix.to_numpy(),
                                        columns=self._numeric_colnames,
                                        index=data.index))
        return imputed

    def _discover_numeric_colnames(self, data):
        """
        Populate self._numeric_colnames with a list containing the names of the
        numeric columns in 'data'.
        """
        self._numeric_colnames = []

        for vname in data.columns:
            maybe_numeric = is_numeric_dtype(data.loc[:, vname])
            if maybe_numeric:
                counts = data.loc[:, vname].value_counts()
                if counts.shape[0] > 20:
                    self._numeric_colnames.append(vname)

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

    def _allocate_space(self, niter, xdim, ydim):
        """
        Allocate space to hold MCMC draws.

        Args:
          niter:  The desired number of draws.
          xdim:  The dimension of the predictor variable.
          ydim:  The dimension of the response variable.
        """
        self.coefficient_draws = np.empty((niter, xdim, ydim))
        self.residual_variance_draws = np.empty((niter, ydim, ydim))
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
                self.atom_error_probs[name][i, cluster, :, :] = (
                    self._model.atom_error_probs(cluster, col).to_numpy()
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

    def _default_atom_prior(self, atoms):
        """
        The default prior on which atom is the 'truth' places 90% probability
        on the continuous atom and splits the remainder equally on the discrete
        atoms.  An atom that is a string of 9's is given prior weight 0.
        """
        number_of_atoms = len(atoms)
        ans = np.ones(number_of_atoms + 1)
        if number_of_atoms > 0:
            ans *= .1 / number_of_atoms
            ans[-1] = .9
        for i in range(len(atoms)):
            atom_int = str(int(atoms[i]))
            if atom_int.startswith("999") and atom_int.endswith("999"):
                ans[i] = -1
        return ans * 100

    def _default_atom_error_prior(self, number_of_atoms):
        """
        The default prior on the conditional distribution of which atom is
        observed, given which atom is true.

        For discrete atoms the prior is that each atom is either observed
        correctly or marked missing.  For the continuous atom the prior places
        equal weight on all categories.

        """
        ans = np.empty((number_of_atoms + 1, number_of_atoms + 2))
        for k in range(number_of_atoms):
            ans[k, :] = -1.0
            ans[k, k] = 1.0
            ans[k, -1] = 1.0
            ans[k + 1, :] = 1.0
        return ans

    def _set_prior(self, xdim, ydim):
        """
        Set the prior distribution on the imputation model.  A nearly
        noninformative prior is chosen for the residual variance matrix and
        regression coefficients.  Each cell in in the mixture model is assigned
        an identical prior.

        The default prior for the variables that do not have a prior
        distribution specified is to put 90% probability on the continuous atom
        being the true case, with the remaining probability spread across the
        other atoms.  With all non-continuous atoms, place 50% prior
        probability on each atom being reported correctly, with the remaining
        50% probability that it is reported as missing.  For the continuous
        atom all reporting categories get equal weight.
        """
        if self._model is None:
            raise Exception(
                "_set_prior was called before the model was created.")

        self._model.set_default_regression_prior()
        self._model.set_default_prior_for_mixing_weights()
        for i in range(len(self._numeric_colnames)):
            vname = self._numeric_colnames[i]
            if vname not in self._atom_prior:
                self._atom_prior[vname] = self._default_atom_prior(
                    self._atoms[vname])
            self._model.set_atom_prior(boom.Vector(self._atom_prior[vname]), i)

            if vname not in self._atom_error_prior:
                self._atom_error_prior[vname] = (
                    self._default_atom_error_prior(len(self._atoms[vname]))
                )
            self._model.set_atom_error_prior(boom.Matrix(
                self._atom_error_prior[vname]), i)

    def __getstate__(self):
        """
        Return a dict that can be used to pickle a MixedDataImputer.
        """
        state = {
            "atoms": self._atoms,
            "numeric_colnames": self._numeric_colnames,
            "categorical_colnames": self._categorical_colnames,
            "atom_prior": self._atom_prior,
            "dataset_encoder": self._dataset_encoder,
            "coefficient_draws": self.coefficient_draws,
            "residual_variance_draws": self.residual_variance_draws,
            "atom_probs": self.atom_probs,
            "num_clusters": self.nclusters,
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
                and state["num_clusters"] is not None
        ):
            xdim = self._encoder.dim
            atom_vector = []
            for vname in self._numeric_colnames:
                atoms = np.array(self._atoms[vname])
                atom_vector.append(boom.Vector(
                    atoms.flatten().astype("float")))
            self._model = boom.MixedDataImputer(
                # TODO: this is hosed.
            )
            state["num_clusters"], atom_vector, xdim
            self._model.set_empirical_distributions(
                state["empirical_distributions"])
