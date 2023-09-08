import numpy as np
import pandas as pd
import scipy.stats


class AnovaTable:
    def __init__(self, sample_sd, SSE, sample_size, xdim):
        self._SSE = SSE
        self._SST = (sample_sd ** 2) * (sample_size - 1)
        self._sample_size = sample_size
        self._xdim = xdim

    @property
    def SSM(self):
        return self._SST - self._SSE

    @property
    def SSE(self):
        return self._SSE

    @property
    def SST(self):
        return self._SST

    @property
    def MSM(self):
        return self.SSM / self.modelDF

    @property
    def MSE(self):
        return self.SSE / self.residualDF

    @property
    def residual_sd(self):
        return np.sqrt(self.SSE / self.residualDF)

    @property
    def sample_sd(self):
        return np.sqrt(self.SST / self.totalDF)

    @property
    def modelDF(self):
        return self._xdim - 1  # assume X contains an intercept

    @property
    def totalDF(self):
        return self._sample_size - 1

    @property
    def residualDF(self):
        return self._sample_size - self._xdim

    @property
    def F(self):
        numerator = self.SSM / self.modelDF
        denominator = self.SSE / self.residualDF
        return numerator / denominator

    @property
    def pvalue(self):
        return scipy.stats.f.sf(self.F, self.modelDF, self.residualDF)

    @property
    def Rsquare(self):
        return 1 - self.SSE / self.SST

    def __repr__(self):
        return f"""
           {"Sum of Sq":10}      {"DF"    :6}   {"Mean SQ   ":10}     {"F-stat":8}   p-value
Model      {self.SSM :10.2}      {self.modelDF   :6}   {self.MSM  :10.4}     {self.F:8.4}   {self.pvalue:8.4}
Error      {self.SSE :10.2}      {self.residualDF:6}   {self.MSE  :10.4}
           ----------      ------
Total      {self.SST :10.2}      {self.totalDF   :6}
"""


class LinearModel:
    """
    A linear regresssion model, because there's not enough lineaer regression
    models in the python universe.

    TODO: robustify this with better matrix decompositions, handle missing data,
    add tests, diagonostics, plots, etc.
    """

    def __init__(self, y, X):
        X = np.array(X)
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        y = np.array(y).ravel()

        xtx = X.T @ X
        xty = X.T @ y
        beta = np.linalg.solve(xtx, xty).ravel()
        yhat = (X @ beta).ravel()
        residuals = y - yhat
        sample_size = len(y)
        sample_sd = np.std(y, ddof=1)
        SSE = residuals.T @ residuals

        self._beta = beta
        self._X = X
        self._y = y
        self._residuals = residuals
        self._anova_table = AnovaTable(
            sample_sd, SSE, sample_size, X.shape[1])
        self._beta_variance = np.linalg.inv(xtx) * self.residual_variance

    @property
    def coef(self):
        return self._beta

    @property
    def coefficients(self):
        return self._beta

    @property
    def residuals(self):
        return self._residuals

    @property
    def residual_df(self):
        return len(self._residuals) - len(self._beta)

    @property
    def residual_sd(self):
        return self._anova_table.residual_sd

    @property
    def residual_variance(self):
        return self.residual_sd ** 2

    @property
    def Rsquare(self):
        return self._anova_table.Rsquare

    @property
    def coefficient_table(self):
        se = np.sqrt(np.diagonal(self._beta_variance))
        tstat = self.coef / se
        pvalue = 2 * scipy.stats.t.sf(np.abs(tstat), df=self.residual_df)
        ans = pd.DataFrame(
            {
                "coef": self.coef,
                "se": se,
                "T": tstat,
                "pvalue": pvalue
            }
        )
        return ans

    @property
    def anova_table(self):
        return self._anova_table


def lm(y, X):
    # TODO: add patsy style formulas, plots, diagnostics, etc.
    return LinearModel(y, X)
