import BayesBoom.boom as boom
import BayesBoom.R as R
import numpy as np
from abc import ABC, abstractmethod


class MeanFunction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def boom(self):
        """
        Produce the corresponding boom object from this wrapper.
        """
        pass

    @abstractmethod
    def create_sampler(self, boom_gp_model):
        pass

    @abstractmethod
    def allocate_space(self, niter: int):
        pass

    @abstractmethod
    def record_draw(self, boom_mean_function, iteration: int):
        pass


class ZeroFunction(MeanFunction):
    def boom(self):
        return boom.ZeroFunction()

    def allocate_space(self, niter: int):
        pass

    def record_draw(self, boom_mean_function, iteration: int):
        pass

    def create_sampler(self, boom_gp_model):
        return boom.GpNullSampler()


class LinearMeanFunction(MeanFunction):
    def __init__(self, coefficients, prior: R.MvnBase):
        self._coefficients = np.array(coefficients).ravel()
        if not isinstance(prior, R.MvnBase):
            raise Exception("prior distribution must be an object inheriting "
                            "from R.MvnBase.")
        if prior.dim != len(self._coefficients):
            raise Exception(
                f"coefficients are of length {len(self._coefficients)} but "
                f"the prior is of dimension {prior.dim}.")
        self._prior = prior
        self._draws = None

    def boom(self):
        return boom.LinearMeanFunction(R.to_boom_vector(self._coefficients))

    def allocate_space(self, niter):
        dim = len(self._coefficients)
        self._coefficient_draws = np.empty((niter, dim))

    def record_draw(self, boom_mean_function, iteration: int):
        self._coefficient_draws[iteration, :] = (
            R.to_numpy(boom_mean_function.coefficients)
        )

    def create_sampler(self, boom_gp_model):
        boom_mean_function = boom_gp_model.mean_function
        return boom.LinearMeanFunctionSampler(
            boom_mean_function,
            boom_gp_model,
            self._prior.boom())
