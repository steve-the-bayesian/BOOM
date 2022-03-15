import BayesBoom.boom as boom
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype, is_categorical_dtype, is_object_dtype, is_bool_dtype
)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import time

from .data_table import to_data_table, to_data_frame


class AutoClean:
    """
    A model for filling in missing values and correcting errors in a data
    table.

    NOTE: This is experimental code.  Its reliability is by no means guaranteed.
    """

    def __init__(self):
        self._dtypes = None
        self.coefficients = None
        self.residual_variance = None
        self.atom_probs = None
        self.atom_error_probs = None
        self.level_probs = None
        self.level_observation_probs = None
        self.mixing_distribution = None
        self._numeric_colnames = None
        self._categorical_colnames = None
        self._atoms = None
        self._atom_prior = {}
        self._atom_error_prior = {}
        self._level_prior = {}
        self._level_observation_prior = {}

    def __getstate__(self):
        state = {
            "dtypes": self._dtypes,
            "coefficients": self.coefficients,
            "residual_variance": self.residual_variance,
            "atom_probs": self.atom_probs,
            "atom_error_probs": self.atom_error_probs,
            "level_probs": self.level_probs,
            "level_observation_probs": self.level_observation_probs,
            "mixing_distribution": self.mixing_distribution,
            "numeric_colnames": self._numeric_colnames,
            "categorical_colnames": self._categorical_colnames,
            "atoms": self._atoms,
            "atom_prior": self._atom_prior,
            "atom_error_prior": self._atom_error_prior,
            "levels": self._levels,
            "level_prior": self._level_prior,
            "level_observation_prior": self._level_observation_prior,
        }
        return state

    def __setstate__(self, state):
        self._dtypes = state["dtypes"]
        self.coefficients = state["coefficients"]
        self.residual_variance = state["residual_variance"]
        self.atom_probs = state["atom_probs"]
        self.atom_error_probs = state["atom_error_probs"]
        self.level_probs = state["level_probs"]
        self.level_observation_probs = state["level_observation_probs"]
        self.mixing_distribution = state["mixing_distribution"]
        self._numeric_colnames = state["numeric_colnames"]
        self._categorical_colnames = state["categorical_colnames"]
        self._atoms = state["atoms"]
        self._atom_prior = state["atom_prior"]
        self._atom_error_prior = state["atom_error_prior"]
        self._level_prior = state["level_prior"]
        self._level_observation_prior = state["level_observation_prior"]
        self._levels = state["levels"]

    def save(self, filename):
        """
        Save the object to a pickle file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        Load a previously fit model from a pickle file.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)   # nosec

    @property
    def nclusters(self):
        if self._model is None:
            return 0
        return self._model.nclusters

    def train_model(self, data: pd.DataFrame, nclusters: int, niter: int,
                    ping: int = 0, checkpoint_filename: str = ""):
        self._start_time = time.time()
        if self._dtypes is None:
            self._dtypes = data.dtypes

        if self._atoms is None:
            self._numeric_colnames = []
            self._categorical_colnames = []
            self._atoms, self._levels = self.find_atoms(data)

        atoms_arg = [boom.Vector(np.array(self._atoms[vname]))
                     for vname in self._numeric_colnames]
        table_arg = to_data_table(data)
        print("creating model object")
        self._model = boom.MixedDataImputerWithErrorCorrection(
            nclusters, table_arg, atoms_arg, boom.GlobalRng.rng)

        print("setting prior")
        self._set_prior()

        print("allocating space")
        self._allocate_space(niter)

        print("about to start mcmc")
        for i in range(niter):
            if (ping > 0) and (i % ping == 0):
                sep = "=-=-=-=-=-=-=-=-="
                print(f"{sep} {time.asctime()} Iteration {i} {sep}")
            self._model.sample_posterior()
            self._record_draws(i)
            if i % niter == 0 and checkpoint_filename != "":
                self.save(checkpoint_filename)

        self._end_time = time.time()

    def find_atoms(self, data: pd.DataFrame):
        """
        Find the numeric atoms and categorical levels to be modeled.
        """
        self._dtypes = data.dtypes
        atoms_dict = {}
        levels_dict = {}
        for i in range(data.shape[1]):
            vname = data.columns[i]
            dt = self._dtypes[i]

            if is_numeric_dtype(dt):
                variable = data.iloc[:, i]
                counts = variable.value_counts().sort_values(ascending=False)
                number_observed = counts.sum()
                atom_indicator = counts > 0.05 * number_observed
                atoms = counts[atom_indicator].index.tolist()
                if len(atoms) > 3:
                    atoms = atoms[:3]
                atoms_dict[vname] = atoms
                self._numeric_colnames.append(data.columns[i])

            elif (
                    is_categorical_dtype(dt)
                    or is_object_dtype(dt)
                    or is_bool_dtype(dt)
            ):
                # TODO: put in some cardinality protections.
                levels = data.iloc[:, i].value_counts()
                levels_dict[vname] = levels
                self._categorical_colnames.append(data.columns[i])

            else:
                raise Exception(
                    "Only categorical or numeric types are supported.")

        return atoms_dict, levels_dict

    def set_atom_prior(self, variable_name: str, prior_counts: np.ndarray):
        """
        Set a constrained (0's are allowed) Dirichlet prior distribution for the
        frequency of "true values" for a numeric variable.

        Args:
          variable_name: The name of the numeric variable that the prior
            describes.
          prior_counts: A 1-d array containing the prior counts.  If the
            variable has k atoms then the array needs k+1 elements, with the
            last one corresponding to the continuous portion of the model.
            Non-positive entries in 'prior_counts' signal that the
            corresponding element has zero probability of being the true value.
        """
        self._atom_prior[variable_name] = prior_counts.ravel()
        if np.sum(prior_counts > 0) == 0:
            raise Exception("At least one prior count must be positive.")

    def set_atom_error_prior(self,
                             variable_name: str,
                             prior_counts: np.ndarray):
        """
        Set a prior distribution on the conditional probability of observing an
        atom in a numeric field, given the true value.

        Args:
          variable_name: The name of the numeric varible described by the
            prior.
          prior_counts: A 2-d numpy array.  If the variable has k atoms then a
            k+1 by k+2 array is needed.  See details below.

        Each row of the matrix
        """
        self._atom_error_prior[variable_name] = prior_counts.astype("float")
        if len(prior_counts.shape) != 2:
            raise Exception("Expected a 2-d array.")

        row_totals = np.sum(prior_counts > 0, axis=1)
        if not np.all(row_totals > 0):
            raise Exception("Each row must have at least one positive entry")

    def set_level_prior(self,
                        variable_name: str,
                        prior_counts: np.ndarray):
        """
        Set a constrained Dirichlet (0 probabilities are allowed) prior
        distribution on the true levels of a categorical variable.

        Args:
          variable_name: The name of the categorical variable described by the
            prior.
          prior_counts: An array of "prior counts".  Non-positive entries
            indicate prior certainty that the corresponding element is zero.
            The array must have dimension matching the number of levels in the
            variable, and at least one entry must be positive.
        """
        self._level_prior[variable_name] = prior_counts.astype(float).ravel()
        if (np.sum(prior_counts > 0) == 0):
            raise Exception("At least one entry must be positive.")

    def set_level_observation_prior(self,
                                    variable_name: str,
                                    prior_counts: np.ndarray):
        """
        Set a collection of independent constrained (0 probabilities are
        allowed) Dirichlet priors on the conditional probability of the
        observed value given true level.

        Args:
          variable_name: The name of the categorical variable described by the
            prior.
          prior_counts: A 2-d array of "prior counts".  Non-positive entries
            indicate prior certainty that the corresponding element is zero.
            Rows correspond to true level values, and columns correspond to
            actual observations.  The number of rows must therefore match the
            number of levels in the variable.  There is one addtional column
            corresponding to "missing".  Each row of the matrix is treated
            independently, and each row must have at least one positive
            element.
        """
        self._level_observation_prior[variable_name] = (
            prior_counts.astype("float")
        )
        if len(prior_counts.shape) != 2:
            raise Exception("A 2-d array is needed.")
        if prior_counts.shape[1] != prior_counts.shape[0] + 1:
            raise Exception("The matrix needs one more column than rows.")
        positive = prior_counts > 0
        row_ok = positive.sum(axis=1)
        if not np.all(row_ok):
            raise Exception("Each row must have at least one positive element.")

    def impute_rows(self, data, iterations):
        """
        Args:
          data:  Data frame containing the rows to impute.
          iterations:  An array-like collection of iteration numbers to use.

        Returns:
          imputed: A list of imputed data frames.  There is one entry for each
            element in 'iterations'
        """
        imputed = []

        data_table = to_data_table(data)

        for it in iterations:
            self._restore_parameters(it)
            imputed_data_table = self._model.impute_data_set(data_table, 10)
            imputed.append(
                to_data_frame(imputed_data_table, columns=data.columns,
                              index=data.index)
            )
        return imputed

    def plot_row_distribution(self, imputations, row, logscale=True, xlim=None):

        imp = pd.concat([frame.iloc[row, :] for frame in imputations])
        numeric_frame = imp.loc[:, self._numeric_colnames]
        if logscale and self._numeric_colnames:
            numeric_frame = np.log1p(numeric_frame)

        fig, ax = plt.subplots(1, figsize=(10, 8))
        g = sns.boxplot(y="variable", x="value", data=numeric_frame, orient="h")
        xlab = "Value"
        if logscale:
            xlab += " (log1p scale)"
        g.set_xlabel(xlab)
        g.set_title(f"Imputation distribution for row {row+1}.")
        if xlim is not None:
            g.set_xlim(xlim)

    def _allocate_space(self, niter: int):
        xdim = self._model.xdim
        ydim = self._model.ydim
        nclusters = self.nclusters

        self.coefficients = np.empty((niter, xdim, ydim))
        self.residual_variance = np.empty((niter, ydim, ydim))

        self.mixing_distribution = np.empty((niter, nclusters))

        self.atom_probs = {}
        self.atom_error_probs = {}
        self.level_probs = {}
        self.level_observation_probs = {}

        for vname in self._numeric_colnames:
            atom_dim = len(self._atoms[vname])
            self.atom_probs[vname] = np.empty((niter, nclusters, atom_dim + 1))
            self.atom_error_probs[vname] = np.empty(
                (niter, nclusters, atom_dim + 1, atom_dim + 2))

        for vname in self._categorical_colnames:
            nlevels = len(self._levels[vname])
            self.level_probs[vname] = np.empty((niter, nclusters, nlevels))
            self.level_observation_probs[vname] = np.empty(
                (niter, nclusters, nlevels, nlevels + 1))

    def _record_draws(self, iteration: int):
        self.coefficients[iteration, :, :] = self._model.coefficients.to_numpy()
        self.residual_variance[iteration, :, :] = (
            self._model.residual_variance.to_numpy()
        )
        self.mixing_distribution[iteration, :] = (
            self._model.mixing_weights.to_numpy()
        )

        for i in range(len(self._numeric_colnames)):
            vname = self._numeric_colnames[i]
            for cluster in range(self.nclusters):
                self.atom_probs[vname][iteration, cluster, :] = (
                    self._model.atom_probs(cluster, i).to_numpy()
                )
                self.atom_error_probs[vname][iteration, cluster, :, :] = (
                    self._model.atom_error_probs(cluster, i).to_numpy()
                )

        it = iteration
        for i in range(len(self._categorical_colnames)):
            vname = self._categorical_colnames[i]
            for cluster in range(self.nclusters):
                self.level_probs[vname][it, cluster, :] = (
                    self._model.level_probs(cluster, i).to_numpy()
                )
                self.level_observation_probs[vname][it, cluster, :, :] = (
                    self._model.level_observation_probs(cluster, i).to_numpy()
                )

    def _restore_parameters(self, iteration: int):
        """
        Restore the state of the model to a specific MCMC iteration.
        """
        self._model.set_coefficients(
            boom.Matrix(self.coefficients[iteration, :, :]))
        self._model.set_residual_variance(
            boom.SpdMatrix(self.residual_variance[iteration, :, :]))
        for cluster in range(self.nclusters):
            for col in range(len(self._numeric_colnames)):
                vname = self._numeric_colnames[col]
                self._model.set_atom_probs(
                    cluster, col,
                    boom.Vector(self.atom_probs[vname][iteration, cluster, :]))
                self._model.set_atom_error_probs(
                    cluster, col,
                    boom.Matrix(self.atom_error_probs[vname][
                        iteration, cluster, :, :]))

            for col in range(len(self._categorical_colnames)):
                vname = self._categorical_colnames[col]
                self._model.set_level_probs(
                    cluster, col,
                    boom.Vector(self.level_probs[vname][iteration, cluster, :]))
                self._model.set_level_observation_probs(
                    cluster, col,
                    boom.Matrix(self.level_observation_probs[vname][
                        iteration, cluster, :, :]))

    def _set_default_regression_prior(self):
        xdim = self._model.xdim
        ydim = self._model.ydim
        b0 = np.zeros((xdim, ydim))
        Sigma = np.diag(np.ones(ydim))
        self._model.set_regression_prior(
            boom.Matrix(b0),
            1.0,
            boom.SpdMatrix(Sigma),
            ydim + 1
        )

    def _set_default_prior_for_mixing_weights(self):
        counts = np.ones(self.nclusters)
        self._model.set_mixing_weight_prior(boom.Vector(counts))

    @staticmethod
    def _default_atom_prior(atoms):
        """
        The default prior on which atom is the 'truth' places 90% probability on
        the continuous atom and splits the remainder equally on the discrete
        atoms.  An atom that is a string of 9's is given prior weight 0.
        """
        number_of_atoms = len(atoms)
        ans = np.ones(number_of_atoms + 1)
        if number_of_atoms > 0:
            ans *= 0.1 / number_of_atoms
            ans[-1] = 0.9
        for i, atom in enumerate(atoms):
            atom_int = str(int(atom))
            if atom_int.startswith("999") and atom_int.endswith("999"):
                ans[i] = -1
        return ans * 100

    @staticmethod
    def _default_atom_error_prior(number_of_atoms):
        """
        The default prior on the conditional distribution of which atom is
        observed, given which atom is true.

        For discrete atoms the prior is that each atom is either observed
        correctly or marked missing.  For the continuous atom the prior places
        equal weight on all categories.

        """
        ans = np.zeros((number_of_atoms + 1, number_of_atoms + 2))
        for k in range(number_of_atoms):
            ans[k, :] = -1.0
            ans[k, k] = 1.0
            ans[k, -1] = 1.0

        ans[number_of_atoms, :] = 1.0
        return ans

    @staticmethod
    def _default_level_prior(levels):
        """
        The default prior is that all levels are equally likely, except for
        all-blank levels which have prior probability 0.

        Args:
          levels:  Array-like list of unique level values.

        Returns:
          A numpy vector of prior counts.
        """
        nlevels = len(levels)
        string_levels = pd.Series(levels, dtype="str").values
        counts = np.ones(nlevels)
        all_blanks = np.array([s.isspace() for s in string_levels])
        counts[all_blanks] = -1
        return counts

    @staticmethod
    def _default_level_observation_prior(levels):
        """
        The default prior splits prior probability equally between the current
        label and the implicit "missing" label in the final column.
        """
        nlevels = len(levels)
        ans = -1 * np.ones((nlevels, nlevels + 1))
        for i in range(nlevels):
            ans[i, i] = 1.0
            ans[i, nlevels] = 1.0
        return ans

    def _set_prior(self):
        """
        Set the prior distribution on the BOOM model object.  If user-specified
        priors have been set using set_atom_prior, set_atom_error_prior, etc
        then those priors will be installed.  Variables for which no prior was
        specified will receive default priors.
        """
        self._set_default_regression_prior()
        self._set_default_prior_for_mixing_weights()

        for i in range(len(self._numeric_colnames)):
            vname = self._numeric_colnames[i]
            if vname not in self._atom_prior:
                self._atom_prior[vname] = self._default_atom_prior(
                    self._atoms[vname])
            self._model.set_atom_prior(boom.Vector(self._atom_prior[vname]), i)

            if vname not in self._atom_error_prior:
                self._atom_error_prior[vname] = self._default_atom_error_prior(
                    len(self._atoms[vname])
                )
            self._model.set_atom_error_prior(boom.Matrix(
                self._atom_error_prior[vname]), i)

        for i in range(len(self._categorical_colnames)):
            vname = self._categorical_colnames[i]
            levels = self._levels[vname]
            if vname not in self._level_prior:
                self._level_prior[vname] = self._default_level_prior(levels)
            self._model.set_level_prior(boom.Vector(np.array(
                self._level_prior[vname])), i)

            if vname not in self._level_observation_prior:
                self._level_observation_prior[vname] = (
                    self._default_level_observation_prior(levels)
                )
            self._model.set_level_observation_prior(
                boom.Matrix(self._level_observation_prior[vname]), i
            )
