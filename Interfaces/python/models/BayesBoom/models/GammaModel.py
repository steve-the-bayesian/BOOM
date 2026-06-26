import numpy as np

from .BaseModels import DoubleModel

class GammaModel(DoubleModel):
    """
    Gamma distribution parameterized by shape (a) and scale (b).

    Mean = a / b,  Variance = a / b^2.

    Can also be constructed from mean (mu) and shape (a) by passing mu and
    shape, or from mu and scale (b).  Exactly two of {shape, scale, mu} must
    be given.
    """

    def __init__(self, shape=None, scale=None, mu=None, a=None, b=None):
        if a is not None:
            shape = a
        if b is not None:
            scale = b

        if (shape is None) + (scale is None) + (mu is None) > 1:
            raise Exception("Exactly two of {shape, scale, mu} must be given.")

        self._a = shape
        if self._a is None:
            self._a = scale * mu

        self._b = scale
        if self._b is None:
            self._b = mu / shape

        if self._a <= 0 or self._b <= 0:
            raise Exception("GammaModel parameters must be positive.")

        self._boom_model = None

    @property
    def mean(self):
        self._refresh_params()
        return self._a / self._b

    @property
    def variance(self):
        self._refresh_params()
        return self._a / self._b ** 2

    @property
    def a(self):
        self._refresh_params()
        return self._a

    @property
    def shape(self):
        self._refresh_params()
        return self._a

    @property
    def b(self):
        self._refresh_params()
        return self._b

    @property
    def scale(self):
        self._refresh_params()
        return self._b

    def boom(self):
        if self._boom_model is None:
            import BayesBoom.boom as boom
            self._boom_model = boom.GammaModel(self._a, self._b)
        return self._boom_model

    def allocate_space(self, niter):
        self._a_draws = np.empty(niter)
        self._b_draws = np.empty(niter)

    def record_draw(self, iteration):
        if self._boom_model is not None:
            self._a_draws[iteration] = self._boom_model.alpha()

    def _refresh_params(self):
        if self._boom_model is not None:
            self._a = self._boom_model.a
            self._b = self._boom_model.b

    def __getstate__(self):
        payload = dict(self.__dict__)
        payload["_boom_model"] = self._boom_model is not None
        return payload

    def __setstate__(self, payload):
        self.__dict__ = payload
        if payload["_boom_model"]:
            self._boom_model = None
            self.boom()

    def __repr__(self):
        return (f"A GammaModel with shape = {self.shape} "
                f"and scale = {self.scale}.")



class SdPrior(DoubleModel):
    """
    Prior on a standard deviation sigma via  1/sigma^2 ~ Gamma(a, b)
    where a = sample_size/2 and b = sample_size * sigma_guess^2 / 2.

    Args:
      sigma_guess: Point estimate of sigma.
      sample_size: Prior weight (number of pseudo-observations).
      initial_value: MCMC starting value; defaults to sigma_guess.
      fixed: If True, hold parameter fixed during MCMC.
      upper_limit: Upper bound on sigma (default: infinity).
    """

    def __init__(self, sigma_guess, sample_size=.01, initial_value=None,
                 fixed=False, upper_limit=np.inf):
        self.sigma_guess = float(sigma_guess)
        self.sample_size = float(sample_size)
        self.initial_value = float(sigma_guess if initial_value is None
                                   else initial_value)
        self.fixed = bool(fixed)
        self.upper_limit = float(upper_limit)

    @property
    def sum_of_squares(self):
        return self.sigma_guess ** 2 * self.sample_size

    def create_chisq_model(self):
        return self.boom()

    def boom(self):
        import BayesBoom.boom as boom
        return boom.ChisqModel(self.sample_size, self.sigma_guess)

    @property
    def mean(self):
        """Mean of the precision (1/sigma^2) distribution."""
        return self.sample_size / self.sigma_guess ** 2

    def __repr__(self):
        return (f"SdPrior with sigma_guess = {self.sigma_guess}, "
                f"sample_size = {self.sample_size}, "
                f"upper_limit = {self.upper_limit}")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, payload):
        self.__dict__ = payload
