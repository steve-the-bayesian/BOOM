import numpy as np
import pandas as pd

import BayesBoom.boom as boom
import BayesBoom.R as R
import BayesBoom.spikeslab as spikeslab

import matplotlib.pyplot as plt
import patsy


class GaussianProcessRegression:
    """
    A Gaussian process regression model.

    Expected usage:

    import BayesBoom.bayesreg as bayesreg
    data = get_some_data_frame()

    formula = "y ~ x1 + x2 + x3"
    X = patsy.dmatrix(formula, data)
    kernel = bayesreg.MahalanobisKernel(X, 1.0)
    mean_function = bayesreg.ZeroFunction()

    model = GaussianProcessRegression(
        bayesreg.ZeroFunction(),
        bayesreg.MahalanobisKernel(X, 1.0),
        1.2)
    model.add_data(data["y"], X)

    model.mcmc(niter=100)
    """

    def __init__(self, mean_function, kernel, residual_sd: float = 1.0):
        self._mean_function = mean_function
        self._kernel = kernel
        self._initial_residual_sd = float(residual_sd)

        self._boom_model = None
        self._residual_sd_prior = None
        self._X = None
        self._y = None

    def set_prior(self, prior: R.SdPrior):
        self._residual_sd_prior = prior

    def add_data(self, response: np.ndarray, predictors: np.ndarray):
        """
        Args:
          predictors: A matrix of predictors.
          response: A vector of responses.  The number of rows in 'predictors'
            must match the length of 'response'.
        """
        self._X = predictors
        self._y = response

    def mcmc(self, niter: int, ping: int = 100):
        self.boom()
        self.allocate_space(niter)
        for iteration in range(niter):
            R.print_timestamp(iteration, ping)
            self._boom_model.sample_posterior()
            self._record_draws(iteration)

    def boom(self):
        self._boom_model = boom.GaussianProcessRegressionModel(
            self._mean_function.boom(),
            self._kernel.boom(),
            self._initial_residual_sd)

        if self._X is not None:
            self._boom_model.add_data(R.to_boom_matrix(self._X),
                                      R.to_boom_vector(self._y))
        self._assign_samplers()
        return self._boom_model

    def create_sampler(self, boom_model):
        """
        Create a boom.PosteriorSampler object suitable for the boom_model, but do
        not assign it.
        """
        kernel_sampler = self._kernel.create_sampler(boom_model)
        mean_function_sampler = self._mean_function.create_sampler(boom_model)
        if self._residual_sd_prior is None:
            self._residual_sd_prior = R.SdPrior(
                .5 * np.std(self._y, ddof=1))

        sampler = boom.GaussianProcessRegressionPosteriorSampler(
            boom_model,
            mean_function_sampler,
            kernel_sampler,
            self._residual_sd_prior.boom(),
            boom.GlobalRng.rng)
        return sampler

    def _assign_samplers(self):
        sampler = self.create_sampler(self._boom_model)
        self._boom_model.set_method(sampler)

    def allocate_space(self, niter: int):
        self._mean_function.allocate_space(niter)
        self._kernel.allocate_space(niter)
        self._residual_sd_draws = np.empty(niter)

    def record_draws(self, iteration):
        self._kernel.record_draw(self._boom_model.kernel, iteration)
        self._mean_function.record_draw(
            self._boom_model.mean_function, iteration)
        self._residual_sd_draws[iteration] = self._boom_model.residual_sd
