from .BaseModels import DoubleModel


class UniformPrior(DoubleModel):
    """Univariate uniform distribution over [lo, hi]."""

    def __init__(self, lo, hi):
        if hi < lo:
            lo, hi = hi, lo
        self._lo = lo
        self._hi = hi

    @property
    def mean(self):
        return .5 * (self._lo + self._hi)

    def boom(self):
        import BayesBoom.boom as boom
        return boom.UniformModel(self._lo, self._hi)


UniformModel = UniformPrior

