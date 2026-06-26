from abc import ABC, abstractmethod
import numpy as np

from .boom_utils import (
    to_boom_vector,
    to_boom_matrix,
    to_boom_spd,
)

# ---------------------------------------------------------------------------
# Multivariate normal models
# ---------------------------------------------------------------------------

class MvnBase(ABC):
    """Abstract base for multivariate normal distributions."""

    @property
    @abstractmethod
    def dim(self):
        """Dimension of the distribution."""

    @property
    @abstractmethod
    def mean(self):
        """Mean vector as a numpy array."""

    @property
    @abstractmethod
    def variance(self):
        """Variance matrix as a 2-d numpy array."""

    @abstractmethod
    def boom(self):
        """Return the corresponding boom object."""

    def create_boom_data_builder(self, data=None):
        from BayesBoom.R.boom_data_builders import VectorDataBuilder
        return VectorDataBuilder()


class MvnModel(MvnBase):
    """Multivariate normal distribution with fixed mean and variance."""

    def __init__(self, mu, Sigma):
        mu = np.asarray(mu)
        Sigma = np.asarray(Sigma)
        if mu.ndim != 1:
            raise Exception("mu must be a vector.")
        if Sigma.ndim != 2:
            raise Exception("Sigma must be a matrix.")
        if Sigma.shape[0] != Sigma.shape[1]:
            raise Exception("Sigma must be square.")
        if Sigma.shape[0] != len(mu):
            raise Exception("mu and Sigma must have matching dimensions.")
        self._mu = mu
        self._Sigma = Sigma
        self._boom_model = None

    @property
    def dim(self):
        return len(self._mu)

    @property
    def mu(self):
        return self._mu

    @property
    def mean(self):
        return self.mu

    @property
    def Sigma(self):
        return self._Sigma

    @property
    def variance(self):
        return self.Sigma

    def boom(self):
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.MvnModel(
                to_boom_vector(self._mu),
                to_boom_spd(self._Sigma))
        return self._boom_model


class MvnGivenSigma(MvnBase):
    """
    Conditional MVN prior given an external variance matrix Sigma.

    Models  y ~ Mvn(mu, Sigma / kappa)  where kappa is 'sample_size'.
    """

    def __init__(self, mu: np.ndarray, sample_size: float):
        self._mu = np.array(mu, dtype="float").ravel()
        self._sample_size = float(sample_size)
        self._boom_model = None

    @property
    def dim(self):
        return len(self._mu)

    def boom(self):
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.MvnGivenSigma(
                to_boom_vector(self._mu), self._sample_size)
        return self._boom_model

    @property
    def variance(self):
        raise Exception(
            "MvnGivenSigma requires a Sigma value to compute the variance.")

    @property
    def mean(self):
        return self._mu

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
