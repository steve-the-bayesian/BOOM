from .BaseModels import DoubleModel


class Ar1CoefficientPrior(DoubleModel):
    """Prior distribution on an AR(1) coefficient."""

    def __init__(self,
                 mu: float = 0.0,
                 sigma: float = 1.0,
                 force_stationary: bool = True,
                 force_positive: bool = False,
                 initial_value: float = None):
        """
        Args:
          mu: Prior mean.
          sigma: Prior standard deviation.
          force_stationary: If True, truncate support to (-1, 1).
          force_positive: If True, truncate support to positive values.
          initial_value: MCMC starting value; defaults to mu.
        """
        self.mu = mu
        self.sigma = sigma
        self.force_stationary = force_stationary
        self.force_positive = force_positive
        self.initial_value = mu if initial_value is None else initial_value

    def boom(self):
        import BayesBoom.boom as boom
        return boom.GaussianModel(self.mu, self.sigma)

    @property
    def mean(self):
        return self.mu

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, payload):
        self.__dict__ = payload

