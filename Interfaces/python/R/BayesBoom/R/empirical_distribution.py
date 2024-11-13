import pandas as pd
import numpy as np
from numbers import Number


class ECDF:
    """
    Empirical cumulative distribution function.
    """

    def __init__(self, data, side="right"):
        """
        Create an empirical cumulative distribution function.

        Args:
          data: The data set whose CDF is desired.

          side: If "right" then this is a traditional CDF, returning P(X <= x).
            If "left" then this returns P(X < x).
        """
        if side not in ["right", "left"]:
            raise Exception(
                "'side' argument must be either 'left' or 'right'.")
        y = np.array(data, copy=True)
        if len(y) == 0:
            raise Exception("ECDF of empty data set.")
        missing = np.isnan(y)
        if np.all(missing):
            raise Exception("All data are missing.")

        self._sorted_data = np.sort(y[~missing])
        self._n = float(len(self._sorted_data))
        self._side = side

    def __call__(self, x):
        return self.cdf(x)

    def cdf(self, x):
        """
        The probability that a randomly chosen data point is less than or
        equal to x.
        """
        if isinstance(x, Number):
            x = np.array([x])
        else:
            x = np.array(x)
        missing = np.isnan(x)
        observed = ~missing
        ans = np.full(len(x), np.nan)

        pos = np.searchsorted(self._sorted_data, x[observed], side=self._side)
        ans[observed] = pos / self._n
        return ans


class NumericEmpiricalDistribution:
    """
    An empirical probability distribution describing a numeric random variable.
    The description is parameterized in terms of the CDF, which can be modeled
    using a collection of quantiles, with values between the quantile points
    handled using interpolation.

    This class is functionally very much like ECDF, but it stores much less
    data and is more easily inverted.

    The algorithm here is taken from Chambers, James, Lambert, Vander Weil
    (2006), Statistical Science.  "Monitoring Networked Applications with
    Incremental Quantile Estimation."  The implementation was adapted from the
    BOOM C++ library by Steven L. Scott.
    """

    def __init__(self, data=None, quantiles=None, bufsize: int = 100):
        """
        Args:
          data: An optional initial data set.  More data can be added later by
            calling add_data.

          quantiles: An array of numbers between 0 and 1 describing points of
            interest on the distribution.

          bufsize: The number of data points to be stored in between model
            refreshes.
        """

        if quantiles is None:
            self._probs = np.array(
                [
                    0,
                    0.01,
                    0.025,
                    0.05,
                    0.10,
                    0.15,
                    0.20,
                    0.25,
                    0.30,
                    0.35,
                    0.40,
                    0.45,
                    0.50,
                    0.55,
                    0.60,
                    0.65,
                    0.70,
                    0.75,
                    0.80,
                    0.85,
                    0.90,
                    0.95,
                    0.975,
                    0.99,
                    1,
                ]
            )
        else:
            self._probs = np.array(list(set(quantiles)), dtype=float)

        self._quantiles = np.zeros_like(self._probs)

        # The empirical CDF will be updated when self._data_buffer grows larger
        # than self._bufsize.
        self._data_buffer = []
        self._bufsize = bufsize

        # The number of observations the CDF has seen.
        self._nobs = 0

        if data is not None:
            self.add_data(data)
            self.update_cdf()

    @classmethod
    def from_summary(self, summary):
        """
        Build a NumericEmpiricalDistribution from a NumericSummary.  This
        allows a an interpolated CDF to be constructed from the quantiles
        in the summary.

        Args:
          summary: A NumericSummary describing the variable.  The quantiles of
            this summary are used to seed the empirical distribution underlying
            this object.
        """
        ans = NumericEmpiricalDistribution()

        dist = summary.frequency_distribution
        if dist is not None:
            probs = np.cumsum(summary.frequency_distribution.values)
            ans._quantiles = np.array(summary.frequency_distribution.index)
            ans._probs = probs / probs[-1]
            ans._nobs = summary.number_observed

        else:
            quantiles = summary.quantiles
            ans._probs = quantiles.index
            ans._quantiles = quantiles.values
            ans._nobs = summary.number_observed
        return ans

    def __repr__(self):
        return f"""
        A NumericEmpiricalDistribution based on {self._nobs} observations,
        with estimated quantiles: \n
        {pd.Series(self._quantiles, index=self._probs)}
        """

    def __call__(self, x):
        """
        The cumulative distribution function evaluated at x.
        """
        return self.cdf(x)

    def add_data(self, x):
        """
        Add data to the model.  Adding data may trigger a model update.

        Args:
          x: float, or array-like.  The data to add to the model.  An array is
            treated like calling add() on each element in the array.  Nan's and
            infinities are ignored.
        """
        if isinstance(x, Number):
            if np.isfinite(x):
                self._data_buffer.append(float(x))
        else:
            x = np.array(x)
            x = x[np.isfinite(x)]
            self._data_buffer += x.tolist()

        if len(self._data_buffer) > self._bufsize:
            self.update_cdf()

    def quantile(self, prob: float):
        """
        A specific quantile of the numeric distribution.  This is the
        inverse-cdf.

        Args:
          prob: The argument to the inverse CDF.
        """
        # TODO(steve): add an exponential or normal tail.
        prob = np.array(prob, dtype=float)
        ans = np.zeros_like(prob)
        ans[prob >= self._probs[-1]] = self._quantiles[-1]
        ans[prob <= self._probs[0]] = self._quantiles[0]

        index = np.logical_and(prob > self._probs[0], prob < self._probs[-1])
        hi = np.searchsorted(self._probs, prob[index], side="left")
        lo = hi - 1
        ans[index] = self._interp(
            prob[index], self._probs[lo], self._probs[hi], self._quantiles[lo],
            self._quantiles[hi]
        )
        return ans

    def cdf(self, x, side="right"):
        """
        The empirical cdf based on the stored quantiles.  The probability that
        the random variable is less than or equal to x.  This calculation does
        not depend on any data in the data buffer.

        Args:
          x: Scalar numeric, or array-like.

        Returns:
          cdf: If the input was a scalar, the output will be as well.  Otherwise
            the output is a numpy array.
        """
        scalar = False
        if isinstance(x, Number):
            x = [x]
            scalar = True
        x = np.array(x, dtype=float)
        ans = np.zeros_like(x)
        missing = np.isnan(x)
        ans[missing] = np.nan
        if np.all(missing):
            return ans

        xobs = x[~missing]
        ansobs = ans[~missing]

        ansobs[xobs < self._quantiles[0]] = 0.0
        ansobs[xobs >= self._quantiles[-1]] = 1.0
        ans[~missing] = ansobs

        index = np.full(x.shape, False)
        index[~missing] = np.logical_and(
            x[~missing] >= self._quantiles[0], x[~missing] < self._quantiles[-1]
        )

        pos = np.searchsorted(self._quantiles, x[index], side=side)
        # This means quantiles[pos-1] <= x < quantiles[pos].  We're assuredly
        # in a situation where quantiles[pos-1] exists, because everything in
        # x[index] is greater than self._quantiles[0].

        plo = self._padded_median(self._probs[pos - 1], self._nobs)
        phi = self._padded_median(self._probs[pos], self._nobs)
        ans[index] = self._interp(
            x[index], self._quantiles[pos - 1], self._quantiles[pos], plo, phi
        )
        if scalar:
            ans = ans[0]
        return ans

    def update_cdf(self):
        """
        Clear the data buffer and update the internal representation.

        The algorithm here is taken from Chambers, James, Lambert, Vander Weil
        (2006), Statistical Science.  "Monitoring Networked Applications with
        Incremental Quantile Estimation."  The notation from that paper is used
        below, with comments explaining the meaning of the mathematical
        symbols.
        """
        if not self._data_buffer:
            return

        # ecdf is a regular cdf: P(X <= x)
        ecdf = ECDF(self._data_buffer)

        # empirical_sub_cdf does not include the equal sign. P(X < x)
        empirical_sub_cdf = ECDF(self._data_buffer, side="left")

        # Get the buffer size here, before it gets augmented by the quantiles.
        bufsize = len(self._data_buffer)

        # Augment the stored data buffer by including the stored quantiles.
        # Keep things sorted.  Only keep unique values.  It is important that
        # the ECDF's and buffer size get computed before this step.
        if self._nobs > 0:
            data_buffer = np.array(
                sorted(set(self._data_buffer + self._quantiles.tolist())),
                dtype=float
            )
        else:
            data_buffer = np.array(sorted(set(self._data_buffer)), dtype=float)

        if self._nobs > 0:
            current_cdf_estimate = self.cdf(data_buffer)
        else:
            current_cdf_estimate = np.zeros_like(data_buffer)

        # Fplus and Fminus weighted averages of the current cdf with the
        # empirical cdf and empirical sub cdf.  They are evaluated at all the
        # data points in the data buffer, which has been augmented with the
        # previous quantile estimates.
        Fplus = (self._nobs * current_cdf_estimate
                 + bufsize * ecdf(data_buffer)) / (
            self._nobs + bufsize
        )
        Fminus = (self._nobs * current_cdf_estimate
                  + bufsize * empirical_sub_cdf(data_buffer)) / (
            self._nobs + bufsize
        )

        # Use Fplus and Fminus to bracket appropriate quantile values.
        for m in range(len(self._probs)):
            # xplus is the smallest data point in data buffer with Fplus bigger
            # than pm.
            pm = self._probs[m]
            xplus_pos = np.nonzero(Fplus >= pm)[0][0]
            xplus = data_buffer[xplus_pos]

            xminus_pos = min(xplus_pos, np.nonzero(Fminus <= pm)[0][-1])
            xminus = data_buffer[xminus_pos]

            if xplus == xminus or Fplus[xplus_pos] == Fminus[xminus_pos]:
                self._quantiles[m] = xminus
            else:
                rho = (Fplus[xplus_pos] - pm) / (
                    Fplus[xplus_pos] - Fminus[xminus_pos])
                if not 0 <= rho <= 1:
                    raise Exception("rho is out of range")
                self._quantiles[m] = rho * xminus + (1 - rho) * xplus

        self._nobs += bufsize
        self._data_buffer.clear()

    def combine(self, other):
        """
        Combine the empirical CDF estimate from this object with that of another
        NumericEmpiricalDistribution.

        The algorithm here is modified from Chambers, James, Lambert, Vander
        Weil (2006), Statistical Science.  "Monitoring Networked Applications
        with Incremental Quantile Estimation."
        """

        # ecdf is a regular cdf: P(X <= x)
        def ecdf(x):
            return other.cdf(x)
        # ecdf = lambda x: other.cdf(x)

        # empirical_sub_cdf does not include the equal sign. P(X < x)
        def empirical_sub_cdf(x):
            return other.cdf(x, side="left")
        # empirical_sub_cdf = lambda x: other.cdf(x, side="left")

        # Get the buffer size here, before it gets augmented by the quantiles.
        bufsize = other._nobs

        # Augment the stored data buffer by including the stored quantiles.
        # Keep things sorted.  Only keep unique values.  It is important that
        # the ECDF's and buffer size get computed before this step.
        if self._nobs > 0:
            data_buffer = np.array(
                sorted(set(other._quantiles.tolist()
                           + self._quantiles.tolist())), dtype=float
            )
        else:
            data_buffer = np.array(sorted(set(other._quantiles)), dtype=float)

        if self._nobs > 0:
            current_cdf_estimate = self.cdf(data_buffer)
        else:
            current_cdf_estimate = np.zeros_like(data_buffer)

        # Fplus and Fminus weighted averages of the current cdf with the
        # empirical cdf and empirical sub cdf.  They are evaluated at all the
        # data points in the data buffer, which has been augmented with the
        # previous quantile estimates.
        Fplus = (self._nobs * current_cdf_estimate
                 + bufsize * ecdf(data_buffer)) / (
            self._nobs + bufsize
        )
        Fminus = (self._nobs * current_cdf_estimate
                  + bufsize * empirical_sub_cdf(data_buffer)) / (
            self._nobs + bufsize
        )

        # Use Fplus and Fminus to bracket appropriate quantile values.
        for m in range(len(self._probs)):
            # xplus is the smallest data point in data buffer with Fplus bigger
            # than pm.
            pm = self._probs[m]
            xplus_pos = np.nonzero(Fplus >= pm)[0][0]
            xplus = data_buffer[xplus_pos]

            if np.any(Fminus <= pm):
                xminus_pos = min(xplus_pos, np.nonzero(Fminus <= pm)[0][-1])
            else:
                xminus_pos = 0
            xminus = data_buffer[xminus_pos]

            if xplus == xminus or Fplus[xplus_pos] == Fminus[xminus_pos]:
                self._quantiles[m] = xminus
            else:
                rho = (Fplus[xplus_pos] - pm) / (
                    Fplus[xplus_pos] - Fminus[xminus_pos])
                if not 0 <= rho <= 1:
                    raise Exception("rho is out of range")
                self._quantiles[m] = rho * xminus + (1 - rho) * xplus

        self._nobs += bufsize

    def _interp(self, x, x0, x1, p0, p1):
        """
        Linearly interpolate between the points (x0, p0) and (x1, p1).
        None of the values are allowed to be nan.
        """
        x = self._to_numpy(x)
        x0 = self._to_numpy(x0)
        x1 = self._to_numpy(x1)
        p0 = self._to_numpy(p0)
        p1 = self._to_numpy(p1)

        ans = np.zeros_like(x)
        # Handle the subset of x's that are equal.
        equal = np.isclose(x1, x0)

        ans[equal] = x1[equal]
        index = ~equal
        ans[index] = p0[index] + (p1[index] - p0[index]) * (
            x[index] - x0[index]) / (x1[index] - x0[index])
        return ans

    def _to_numpy(self, x):
        if isinstance(x, Number):
            x = [x]
        return np.array(x, dtype=float)

    def _padded_median(self, prob, T):
        """
        Return the median of prob 1/2T, and 1 - 1/2T.  The latter two are the
        lower and upper probability extremes for a data set with T elements.
        """
        prob = np.array(prob, dtype=float)
        lo = 0.5 / T
        hi = 1 - (0.5 / T)
        ans = prob.copy()
        ans[lo > prob] = lo
        ans[hi < prob] = hi
        return ans
