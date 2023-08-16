import BayesBoom.R as R
import BayesBoom.boom as boom
import numpy as np


"""
Python wrappers for C++ objects relating to kernels and mean functions.
"""


class ZeroFunction:
    def boom(self):
        return boom.ZeroFunction()

    def allocate_space(self, niter: int):
        pass

    def record_draw(self, boom_mean_function, iteration: int):
        pass

    def create_sampler(self, boom_model):
        return boom.GpNullSampler()


class MahalanobisKernel:
    def __init__(self,
                 predictor_matrix: np.ndarray,
                 scale: float = 1.0,
                 diagonal_shrinkage: float = 0.05,
                 scale_prior: R.DoubleModel = R.SdPrior(1.0, 1.0)):
        """
        Args:
          predictor_matrix: A matrix
        """
        self._X = predictor_matrix
        self._scale = float(scale)
        self._diagonal_shrinkage = float(diagonal_shrinkage)
        self._scale_prior = scale_prior

    def boom(self):
        return boom.MahalanobisKernel(
            R.to_boom_matrix(self._X),
            self._scale,
            self._diagonal_shrinkage)

    def create_sampler(self, boom_model):
        """
        Args:
          boom_model:  A boom.GaussianProcessRegressionModel object
        """
        if self._scale_prior is not None:
            boom_kernel = boom_model.kernel_param
            sampler = boom.MahalanobisKernelSampler(
                boom_kernel,
                boom_model,
                self._scale_prior.boom())
        else:
            sampler = boom.NullSampler()
        return sampler

    def allocate_space(self, niter: int):
        self._scale_draws = np.empty(niter)

    def record_draw(self, boom_kernel, iteration: int):
        self._scale_draws[iteration] = boom_kernel.scale
