import numpy as np

class WishartModel:
    """
    Wishart prior for the precision parameter of an MVN distribution.

    If X_i ~ Mvn(0, V) independently, then X'X ~ Wishart(nu, V) where nu is
    the degrees of freedom.  Mean of the distribution is nu * V.
    """

    def __init__(self, df: float, variance_estimate: np.ndarray):
        variance_estimate = np.asarray(variance_estimate)
        sumsq = df * variance_estimate
        if sumsq.ndim != 2:
            raise Exception("variance_estimate must be a matrix.")
        if sumsq.shape[0] != sumsq.shape[1]:
            raise Exception("variance_estimate must be square.")
        sym_sumsq = (sumsq + sumsq.T) * .5
        if np.sum(np.abs(sumsq - sym_sumsq)) / np.sum(np.abs(sumsq)) > 1e-8:
            raise Exception("variance_estimate must be symmetric.")
        if df <= sumsq.shape[0]:
            raise Exception(
                "df must be larger than nrow(variance_estimate) for the "
                "prior to be proper.")
        self._df = df
        self._sumsq = sumsq

    @property
    def variance_estimate(self):
        return self._sumsq / self._df

    @property
    def df(self):
        return self._df

    def boom(self):
        import BayesBoom.boom as boom
        return boom.WishartModel(self.df, self.variance_estimate)






