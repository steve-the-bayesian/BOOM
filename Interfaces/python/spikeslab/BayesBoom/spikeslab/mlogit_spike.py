import numpy as np
import pandas as pd
import patsy
import BayesBoom.boom as boom
import BayesBoom.R as R
import scipy.sparse

from .priors import MultinomialLogitSpikeSlabPrior
from .spikeslab import sparsify


class mlogit_spike:
    """
Suppose the data looks like:
country         age     sex     married size    type
American	34	Male	Married	Large	Family
Japanese	36	Male	Single	Small	Sporty
Japanese	23	Male	Married	Small	Family
American	29	Male	Single	Large	Family
American	39	Male	Married	Medium	Family
Japanese	34	Male	Single	Medium	Family

    """

    def __init__(self,
                 response,
                 subject_formula: str,
                 choice_formula: dict,
                 niter: int,
                 data: pd.DataFrame,
                 levels: list = None,
                 prior: MultinomialLogitSpikeSlabPrior = None,
                 ping: int = None,
                 seed: int = None,
                 **kwargs):
        """
        Create and a model object and run a specified number of MCMC iterations.

        Args:
          response: Either a string naming a column in the data frame, or an
            object convertible to a 1-d numpy array of dtype "object".
          subject_formula: A model formula that can be interpreted by 'patsy'
            to produce a model matrix from 'data'.
          choice_formula: A dict of strings, keyed by levels of the response
            variable.  Dictionary entries are strings that can be fed to
            'patsy' to construct predictor matrices from 'data'.  See the
            method build_choice_formula for help constructing the formula
            semi-programmatically.
          niter: The desired number of MCMC iterations.
          data: A pd.DataFrame containing the data with which to train the
            model.
          prior: A SpikeSlabPrior object providing the prior distribution over
            the inclusion indicators, the coefficients, and the residual
            variance parameter.
          ping: The frequency (in iterations) with which to print status
            updates.  If ping is None then niter/10 will be assumed.
          seed: The seed for the C++ random number generator, or None.
          **kwargs: Extra argumnts will be passed to SpikeSlabPrior.

        Returns:
          An lm_spike object.
        """

        if not levels:
            if isinstance(choice_formula, dict) and len(choice_formula) > 0:
                levels = list(choice_formula.keys())
            else:
                levels = list(pd.unique(response))
        self._levels = levels

        subject_predictors = patsy.dmatrix(subject_formula, data, eval_env=1)
        self._subject_x_design_info = subject_predictors.design_info

        choice_predictors = {
            x: patsy.dmatrix(choice_formula[x], data, eval_env=1)
            for x in self._levels
        } if choice_formula else {}
        self._choice_x_design_info = {k: v.design_info
                                      for k, v in choice_predictors}

        # xdim = predictors.shape[1]
        # sample_size = predictors.shape[0]
        niter = int(niter)
        if niter <= 0:
            raise Exception("niter should be a positive integer.")

        if ping is None:
            ping = int(niter / 10)
        ping = int(ping)

        if seed is not None:
            boom.GlobalRng.rng.seed(int(seed))

        self._model = boom.MultinomialLogitModel(
            np.array(response, dtype="str"),
            R.to_boom_matrix(subject_predictors),
            [R.to_boom_matrix(choice_predictors[x] for x in levels)]
            if choice_predictors else [])

        if prior is None:
            prior = MultinomialLogitSpikeSlabPrior.from_model(self._model)
        prior.create_sampler(self._model, assign=True)

        # A lil matrix is a "linked list" matrix.  This is an efficient method
        # for constructing matrices.  It should be converted to a different
        # matrix type before doing anything with it.
        nvars = self._model.beta_size(include_zeros=False)
        self._coefficient_draws = scipy.sparse.lil_matrix((niter, nvars))
        self._log_likelihood = np.zeros(niter)

        for i in range(niter):
            self._model.sample_posterior()
            beta = self._model.coefficients
            self._coefficient_draws[i, :] = sparsify(beta)
            self._log_likelihood[i] = self._model.log_likelihood

        # Convert the coefficient draws to sparse column format.  Predictions
        # vs this format should take the form X @ beta, not beta @ X.
        self._coefficient_draws = self._coefficient_draws.tocsc()

    @property
    def xdim(self):
        return self._model.xdim

    @property
    def log_likelihood(self):
        return self._log_likelihood

    @property
    def xnames(self):
        # A list of strings containing the column names of the predictors.
        return self._x_design_info.column_names

    def predict(self, newdata, burn=None, seed=None):
        """
        Return an LmSpikePrediciton object.
        """
        if burn is None:
            burn = R.suggest_burn(self.log_likelihood)
        if seed is not None:
            boom.GlobalRng.rng.seed(int(seed))
        if isinstance(newdata, np.ndarray) and len(newdata.shape) == 1:
            newdata = newdata.reshape(1, -1)
        if isinstance(newdata, np.ndarray) and newdata.shape[1] == self.xdim:
            predictors = newdata
        else:
            predictors = patsy.build_design_matrices(
                [self._x_design_info],
                data=newdata)[0]
        return self._coefficient_draws[burn:, :] @ predictors.T

    @staticmethod
    def build_choice_formula(base_formula: str, levels, prefix="[X]"):
        """
        A common way to include choice-level predictors in a data frame is via a
        naming scheme like

        A_size, B_size, C_size, A_gas_mileage, B_gas_mileage, C_gas_mileage, ...

        This function allows you to specify a model formula along the lines of
        "[X]_size + [X]_gas_mileage".

        Args:
          base_formula: A string giving the model formula for the 'choice' part
            of the model in terms of a prefix that will be replaced by the
            levels of the response variable.
          levels:  A list containing the levels of the response variable.
          prefix: The dummy string to be used in 'base_formula' in place of the
            level names.

        Returns:
          A dict, keyed by the values in 'levels'.  Each dict entry is a string
            containing 'base_formula' but with 'prefix' replaced by the the
            values in 'levels'.
        """
        return {x: base_formula.replace(prefix, str(x)) for x in levels}
