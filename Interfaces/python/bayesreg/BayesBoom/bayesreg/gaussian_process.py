import numpy as np
import pandas as pd

import BayesBoom.boom as boom
import BayesBoom.R as R
import BayesBoom.spikeslab as spikeslab

import matplotlib.pyplot as plt
import patsy
from datetime import datetime


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
    model.add_data(X, data["y"])


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

    def add_data(self, predictors: np.ndarray, response: np.ndarray):
        """
        Args:
          predictors: A matrix of predictors.
          response: A vector of responses.  The number of rows in 'predictors'
            must match the length of 'response'.
        """
        self._X = predictors
        self._y = response

    def mcmc(self, niter: int, ping: int = 100):
        self._boom_model = boom.GaussianProcessRegressionModel(
            self._mean_function.boom(),
            self._kernel.boom(),
            self._initial_residual_sd)

        if self._X is not None:
            self._boom_model.add_data(R.to_boom_matrix(self._X),
                                      R.to_boom_vector(self._y))
        self._create_samplers()
        self._allocate_space(niter)
        for iteration in range(niter):
            R.print_timestamp(iteration, ping)
            self._boom_model.sample_posterior()
            self._record_draws(iteration)

    def _create_samplers(self):
        kernel_sampler = self._kernel.create_sampler(self._boom_model)
        mean_function_sampler = self._mean_function.create_sampler(
            self._boom_model)
        sampler = boom.GaussianProcessRegressionPosteriorSampler(
            self._boom_model,
            mean_function_sampler,
            kernel_sampler,
            self._residual_sd_prior.boom(),
            boom.GlobalRng.rng)
        self._boom_model.set_method(sampler)

    def _allocate_space(self, niter: int):
        self._mean_function.allocate_space(niter)
        self._kernel.allocate_space(niter)
        self._residual_sd_draws = np.empty(niter)

    def _record_draws(self, iteration):
        self._kernel.record_draw(self._boom_model.kernel_param, iteration)
        self._mean_function.record_draw(self._boom_model.mean_param, iteration)
        self._residual_sd_draws[iteration] = self._boom_model.residual_sd
