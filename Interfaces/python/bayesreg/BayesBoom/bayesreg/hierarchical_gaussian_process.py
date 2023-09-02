import numpy as np
import pandas as pd

import BayesBoom.boom as boom
import BayesBoom.R as R

import copy
# import matplotlib.pyplot as plt
# import patsy

from .gaussian_process import GaussianProcessRegression
from .mean_function import MeanFunction, LinearMeanFunction, ZeroFunction
from .kernels import Kernel, MahalanobisKernel


class HierarchicalGaussianProcessRegression:
    """
    A hierarchical Gaussian process regression model, which is a collection of
    GP regressions that have a GP regression as a 'prior mean function.'.

    Expected usage (with default settings -- fine control over mean functions
    and kernels is possible):

    import BayesBoom.bayesreg as bayesreg
    predictors, response, group_ids = get_data_from_somewhere()
    model = HierarchicalGaussianProcessRegression()
    model.add_data(predictors, response, group_id_strings)
    model.mcmc(niter=1000)
    """
    def __init__(self,
                 mean_function: MeanFunction = None,
                 prior_kernel: Kernel = None,
                 data_kernel: Kernel = None,
                 expected_R2: float = 0.5):
        """
        Args:
          mean_function: The mean function to use for the prior distribution.
            Either None (in which case a LineaerMeanFunction will be used), or
            an object inheriting from MeanFunction.
          prior_kernel: The kernel to use for the prior distribution. Either
            None (in which case a default MahalanobisKernel will be used) or an
            object inheriting from Kernel.
          data_kernel: The kernel to use for the data-level models at the
            leaves of the hierarchy.  If None then the data-level models will
            use the same kernel as the prior distribution.
          expected_R2: Used to set the prior on the residual standard
            deviation.  The prior guess at the residual SD is the sample SD *
            sqrt(1 - expected_R2).  This follows from the linear regression
            relationship where the residual variance = (1 - R**2) * sample
            variance.
        """
        # To be filled with a boom.HierarchicalGpRegressionModel object.
        self._model = None

        # self._prior is the GaussianProcessRegression that serves as a
        # mean_function to all the data models at the leaves of the heirarchy.
        #
        # If we don't have enough information to create self._prior right
        # now, then store self._hyperprior_mean_function and self._prior_kernel
        # (if either is given) for use when self._prior is created.
        # Data-dependent defaults are used for the mean function and kernel if
        # they are passed as None, so we can't set the default values here.
        #
        # In the event that self._prior is constructed later,
        # self._hyperprior_mean_function and self._prior_kernel are removed
        # from the object after self._prior is built.  They are stored in
        # self._prior, so there's no need to also store them here.
        if mean_function is not None and prior_kernel is not None:
            self._prior = GaussianProcessRegression(
                mean_function,
                prior_kernel,
                1.0)
        else:
            self._hyperprior_mean_function = mean_function
            self._prior_kernel = prior_kernel
            self._prior = None

        # Group models is to be filled with bayesreg.GaussianProcessRegression
        # objects describing the model for each group.
        self._group_models = {}

        # When 'assign_data' is called, 'X' is filled with a matrix of
        # predictors, 'y' with a vector of responses, and 'group' with a
        # pd.Categorical indicating the group in the heirarchy to that owns
        # each observation.
        self._X = None
        self._y = None
        self._group = None

        # When group-level data models are added, the kernel is a copy of
        # self._data_kernel_prototype.
        self._data_kernel_prototype = data_kernel

        self._expected_R2 = expected_R2

    def add_data(self, predictors: np.ndarray, response: np.ndarray, group):
        """
        Args:
          predictors: A matrix of predictors.
          response: A vector of responses.  The number of rows in 'predictors'
            must match the length of 'response'.
          group:  A list of strings
        """
        if self._X is None:
            self._X = np.array(predictors)
            self._y = np.array(response).ravel()
            self._group = pd.Categorical(group)

        else:
            self._X = np.concatenate([self._X, np.array(predictors)], axis=0)
            self._y = np.concatenate([self._y, np.array(response).ravel()])
            self._group = pd.Categorical(self._group.to_list() + list(group))

    @property
    def prior(self):
        """
        The shared GP regression serving as a prior mean function for the
        data-level models.
        """
        return self._prior

    @property
    def groups(self):
        """
        The unique set of group labels in the leaves of the hierarchy.
        """
        return list(self._group_models.keys())

    def data_model(self, group):
        """
        The bayesreg.GaussianProcessRegression describing the data in the specified
        group.  This object holds the boom model that is actually used in the
        MCMC algorithm, and it holds the draws of the model parameters that it
        owns.

        The boom model (i.e. the C++ model object) is aware of the data
        assigned to the group, but the python object has no data assigned.
        """
        return self._group_models.get(group, None)

    def mcmc(self, niter: int, ping: int = 100):
        """
        Run the MCMC model fitting algorithm for the specified number of
        iterations.  Print a status update every 'ping' iterations.
        """
        self._boom_model = self._create_model()
        self._allocate_space(niter)
        for iteration in range(niter):
            R.print_timestamp(iteration, ping)
            self._boom_model.sample_posterior()
            self._record_draws(iteration)

    def _create_model(self):
        """
        Create the boom_model object that the calling function will store.  This
        function creates the model, assigns the data, and assigns posterior
        samplers.

        Returns:
          A boom.HierarchicalGpRegressionModel that is ready to support MCMC.
        """
        sdy = np.std(self._y, ddof=1)
        self._prior = self._create_prior_mean_function()
        if self._prior._residual_sd_prior is None:
            self._prior.set_prior(R.SdPrior(
                sdy * np.sqrt(1 - self._expected_R2)))

        model = boom.HierarchicalGpRegressionModel(self._prior.boom())
        self._add_data_models(model)
        if self._X is not None:
            model.add_data(
                R.to_boom_vector(self._y),
                R.to_boom_matrix(self._X),
                self._group
            )
        else:
            raise Exception("Model has no data assigned")

        sampler = boom.HierarchicalGpPosteriorSampler(
            model, boom.GlobalRng.rng)
        model.set_method(sampler)

        return model

    def _add_data_models(self, boom_model: boom.HierarchicalGpRegressionModel):
        """
        Args:
          boom_model: A hierarchical model ready to be decorated with data
            models.
        """
        if self._data_kernel_prototype is None:
            self._data_kernel_prototype = self._prior._kernel

        residual_sd = np.std(self._y, ddof=1) * np.sqrt(1 - self._expected_R2)

        # By setting the mean function parameter to zero_fun, we ensure that no
        # posterior sampler gets created for the mean function.  On the C++
        # side, the hierarchical model will replace the given mean function
        # with the prior.
        zero_fun = ZeroFunction()
        unique_groups = self._group.unique().to_list()
        for group in unique_groups:
            py_data_model = GaussianProcessRegression(
                zero_fun,
                copy.deepcopy(self._data_kernel_prototype),
                residual_sd)
            self._group_models[group] = py_data_model
            py_data_model.set_prior(R.SdPrior(residual_sd))

            #
            boom_data_model = py_data_model.boom()
            boom_model.add_model(boom_data_model, str(group))
        return boom_model

    def _create_prior_mean_function(self):
        """
        Create a boom.GaussianProcessRegressionModel object ready to be
        """
        # Don't do anything if the prior has already been created.
        if self._prior is not None:
            return self._prior

        if self._hyperprior_mean_function is None:
            self._hyperprior_mean_function = (
                self._default_hyperprior_mean_function()
            )

        if self._prior_kernel is None:
            self._prior_kernel = self._create_prior_kernel()

        sample_sd = np.std(self._y, ddof=1)
        residual_sd = sample_sd * np.sqrt(1 - self._expected_R2)

        ans = GaussianProcessRegression(
            self._hyperprior_mean_function,
            self._prior_kernel,
            residual_sd)

        del self._hyperprior_mean_function
        del self._prior_kernel
        return ans

    def _default_hyperprior_mean_function(self):
        """
        Create a bayesreg.LineaerMeanFunction object representing the mean
        function used in the prior model.
        """
        if self._X is None:
            raise Exception("No data has been assigned.")
        reg = R.lm(self._y, self._X)
        prior = R.ScottZellnerMvnPrior(R.RegSuf.from_data(self._X, self._y))
        return LinearMeanFunction(reg.coefficients, prior)

    def _create_prior_kernel(self):
        """
        Create a bayesreg.MahalanobisKernel object representing the kernel
        function used in the prior model.
        """
        return MahalanobisKernel(self._X)

    def _allocate_space(self, niter: int):
        self._prior.allocate_space(niter)
        for data_model in self._group_models.values():
            data_model.allocate_space(niter)

    def _record_draws(self, iteration):
        self._prior.record_draws(iteration)
        for data_model in self._group_models.values():
            data_model.record_draws(iteration)
