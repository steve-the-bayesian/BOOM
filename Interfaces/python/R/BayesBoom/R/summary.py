import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype


def summary(x, max_levels: int = 10, numeric_min_unique: int = 10, **kwargs):
    """
    Return the appropriate categorical or numeric summary.

    Args:
      x: Either a pd.DataFrame, or an array-like variable to be summarized.
      max_levels: The maximum number of levels to return in a categorical
        variable.
      numeric_min_unique: A numeric variable with fewer than this many distinct
        values will be summarized as categorical.
      **kwargs: keyword arguments passed to the appropriate summary object
        (NumericSummary, etc).

    Returns: If x is a data frame, the return is a dict, keyed by column name,
        containing summaries of each column.  Otherwise the return is a summary
        of the argument.
    """
    if isinstance(x, pd.DataFrame):
        ans = {}
        for column in x.columns:
            print(f"summarizing variable: {column}")
            ans[column] = summary(
                x[column],
                max_levels=max_levels,
                numeric_min_unique=numeric_min_unique
            )
        return ans
    else:
        if (
                isinstance(x, np.ndarray)
                and len(x.shape) == 2
                and np.min(x.shape) == 1
        ):
            x = x.reshape(-1)

        x = pd.Series(x, dtype=float)
        if is_numeric_dtype(x.dtype) and x.nunique() >= numeric_min_unique:
            return NumericSummary(x, **kwargs)
        elif is_datetime64_any_dtype(x):
            return DateTimeSummary(x, **kwargs)
        else:
            return CategoricalSummary(x, max_levels, **kwargs)


def pad(series, nspaces=4):
    """
    Format a pd.Series object for output by prepending 'nspaces' empty space
    characters.
    """
    padding = " " * nspaces
    output = padding + str(series)
    output = output.replace("\n", "\n" + padding)
    return output


def granularity(datetime):
    """
    Returns the smallest time interval between non-identical time stamps.
    """
    times = pd.Series(np.sort(datetime.unique()), dtype="datetime64[ns]")
    dt = times.diff()
    zero = pd.Timedelta(0)
    if np.any(dt > zero):
        return np.min(dt[dt > zero])
    else:
        return zero


def granularity_bucket(timedelta: pd.Timedelta):
    """
    Given a pd.Timedelta describing the typical time difference between
    events. Characterize the granularity as a descriptive category.

    Args:
      timedelta:
        A duration thought to be a typical gap between observed events.

    Returns:
      A string indicating the timescale of the data.  Possible values are
      "zero", "seconds" "minutes", "hourly", "daily", "weekly", "monthly",
      "quarterly", "yearly", "unknown"
    """
    dt = pd.Timedelta(1, "s")
    seconds = timedelta / dt
    if abs(seconds - 1.0) < .5:
        return "seconds"
    elif seconds < .5:
        return "zero"
    elif abs(seconds - 60) < 5:
        return "minutes"
    elif abs(seconds - 3600) < 120:
        return "hourly"

    dt = pd.Timedelta(1, "d")
    days = timedelta / dt
    if abs(days - 1.0) < .1:
        return "daily"
    elif abs(days - 7.0) < 1:
        return "weekly"
    elif abs(days - 30.5) < 3:
        return "monthly"
    elif abs(days - 90) < 5:
        return "quarterly"
    elif abs(days - 365) < 5:
        return "yearly"

    else:
        return "unknown"


def is_all_nines(x):
    """
    Check whether x is a positive or negative string of all 9's.  This is a
    common way of encoding missing data.

    Args:
      x:  a number (python scalar)

    Return:
        True if x is "all nines".  False otherwise.
    """
    # If x is infinite, nan, or has a fractional part then we don't consider it
    # "all 9's".
    if not np.isfinite(x) or (x != int(x)):
        return False

    x = str(abs(int(x)))
    return len(x) >= 3 and x[0] == "9" and len(set(x)) == 1


day_names = np.array(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                      "Saturday", "Sunday"])


def weekday(datetime):
    """
    Convert a datetime64 object into the day of the week, expressed as a string.
    """
    datetime = pd.Series(datetime, dtype="datetime64[ns]")
    day_codes = datetime.dt.weekday
    obs = ~day_codes.isna()
    ans = pd.Series(
        pd.Categorical([np.nan] * datetime.shape[0], categories=day_names),
        index=datetime.index
    )
    # The code below works around an issue in numpy / pandas where they fail to
    # gracefully handle the case where only one value is observed.
    nobs = np.sum(obs)
    observed_values = day_names[day_codes[obs].astype(int)]
    if nobs > 1:
        ans[obs] = observed_values
    elif nobs == 1:
        ans[obs] = str(observed_values[0])
    return ans


month_names = np.array([
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"])


def month(datetime):
    """
    Convert a datetime object in the the month of the year, expressed as a
    string.
    """
    datetime = pd.Series(datetime, dtype="datetime64[ns]")
    month_codes = datetime.dt.month - 1
    obs = ~month_codes.isna()
    ans = pd.Series(
        pd.Categorical([np.nan] * datetime.shape[0], categories=month_names),
        index=datetime.index
    )
    # The code below works around an issue in numpy / pandas where they fail to
    # gracefully handle the case where only one value is observed.
    nobs = np.sum(obs)
    observed_values = month_names[month_codes[obs].astype(int)]
    if nobs > 1:
        ans[obs] = observed_values
    elif nobs == 1:
        ans[obs] = str(observed_values[0])
    return ans


class UnivariateSummary(ABC):
    """
    A summary of a single "column" of data.  The concrete type of summary
    depends on the type of data being summarized.  Numbers, categories, and
    dates all require very different handling.
    """

    @property
    @abstractmethod
    def sample_size(self):
        """
        The number of observations being summarized.
        """

    @property
    @abstractmethod
    def number_missing(self):
        """
        The number of missing observations.
        """

    @property
    @abstractmethod
    def number_observed(self):
        """
        The number of non-missing observations.
        """

    @property
    @abstractmethod
    def number_unique(self):
        """
        The number of distinct values observed.
        """


class NumericSummary(UnivariateSummary):

    def __init__(self, x, **kwargs):
        """
        Args:
          x: The numeric variable to be summarized.
        """
        # Calling with x as None should only be done by another constructor.
        if x is not None:
            self._summarize(x, **kwargs)

    def _summarize(self, x, quantiles=None):
        X = pd.Series(x)

        try:
            # Flag potential missing value codes.  These are highly repeated
            # values with lots of 9's in them and no other characters.  Could be
            # positive or negative.
            self._number_observed = X.count()
            self._number_missing = len(X) - self._number_observed
            self._number_unique = X.nunique()
            self._flag_potential_missing_value_codes(X)
            self._missing_values_assumed = False

            # Always treat infinite values as missing.
            if np.isinf(X.max()) or np.isinf(X.min()):
                Z = X.replace([np.inf, -np.inf], np.nan)

                self._summarize(Z)
                return

            if self._number_missing == 0 and len(
                    self._potential_missing_value_codes) == 1:
                # If there are no other missing data, and there appears to be a
                # single missing value code, go ahead and treat the code as
                # missing data.
                missing_code = self._potential_missing_value_codes[0]
                Z = X.replace(missing_code, np.nan)
                self._summarize(Z)
                # The summarize method will obliterate the missing value code.
                # Put it back.
                self._potential_missing_value_codes = [missing_code]
                self._missing_values_assumed = True
                return

            X = X.dropna()

            if quantiles is None:
                deciles = np.arange(11) / 10
                quantiles = np.sort(
                    list(deciles) + [0.25, 0.75, 0.01, 0.99, 0.025, 0.975,
                                     0.05, 0.95]
                )
            self._quantiles = X.quantile(q=quantiles)

            self._mean = X.mean()
            self._sd = X.std(ddof=1)
            self._skew = X.skew()
            self._kurtosis = X.kurt()
            self._outlier_count = ((X - self._mean).abs() / self._sd > 5).sum()

            if self._number_unique < 20:
                self._frequency_distribution = X.value_counts().sort_index()
            else:
                self._frequency_distribution = None

            self._compute_histogram(X)

        except Exception as e:
            message = "Could not summarize a numeric variable with initial "
            message += f"values {X.head()} "
            message += f"and dtype {X.dtype}.\n"
            message += "Original error message: " + str(e)
            raise Exception(message)

    @property
    def sample_size(self):
        return self._number_missing + self._number_observed

    @property
    def number_observed(self):
        return self._number_observed

    @property
    def number_missing(self):
        return self._number_missing

    @property
    def number_unique(self):
        """
        The number of distinct values observed in the data.  If the summary was
        constructed by merging two distributed summaries then this value may be
        a lower bound.
        """
        return self._number_unique

    def cdf(self, x):
        """
        The approximate fraction of the data less than or equal to x.

        Args:
          x: The location where the cdf is to be evaluated.  Vectorized values
             are supported.
        """
        from .empirical_distribution import NumericEmpiricalDistribution
        dist = NumericEmpiricalDistribution.from_summary(self)
        return dist(x)

    def inverse_cdf(self, probs):
        """
        An approximation to the inverse CDF, also known as the quantile
        function. Given a probability p, return the value x such that p% of
        the data is less than x.
        """
        from .empirical_distribution import NumericEmpiricalDistribution
        dist = NumericEmpiricalDistribution.from_summary(self)
        return dist.quantile(probs)

    @property
    def frequency_distribution(self):
        """
        The distribution of the top several numeric values, expressed as counts.
        The frequency distribution is only captured if the number of unique
        values is less than 20.  Otherwise self_frequency distribution is None.
        """
        return self._frequency_distribution

    @property
    def proportion_missing(self):
        total_count = self._number_missing + self._number_observed
        if total_count == 0:
            return 0.0
        else:
            return float(self._number_missing) / total_count

    @property
    def quantiles(self):
        return self._quantiles

    @property
    def min(self):
        return self._quantiles.iloc[0]

    @property
    def max(self):
        return self._quantiles.iloc[-1]

    @property
    def median(self):
        return self._quantiles.loc[0.5]

    @property
    def mean(self):
        return self._mean

    @property
    def sd(self):
        return self._sd

    @property
    def var(self):
        return self._sd ** 2

    @property
    def skew(self):
        return self._skew

    @property
    def kurtosis(self):
        return self._kurtosis

    @property
    def outlier_count(self):
        """
        The number of observations at least 5 standard deviations from the mean.
        """
        return self._outlier_count

    @property
    def potential_missing_value_codes(self):
        """
        A list, which might be empty, containing candidates for numeric codes
        that might have been used to denote missingness.  These are defined in
        _flag_potential_missing_value_codes.
        """
        return self._potential_missing_value_codes

    @property
    def missing_values_assumed(self):
        """
        bool.  If True then the summary was computed by assuming that the unique
        value in potential_missing_value_codes is actually a flag for
        missingness.
        """
        return self._missing_values_assumed

    def _compute_histogram(self, X):
        """
        Computes a histogram for the vector of numeric data X.
        """
        self._histogram = np.histogram(X)
        self._log_histogram = None
        self._log_histogram_add_one = False
        if self.min > 0.0:
            self._log_histogram = np.histogram(np.log10(X))
        elif self.min == 0.0:
            self._log_histogram = np.histogram(np.log10(1 + X))
            self._log_histogram_add_one = True

    def _flag_potential_missing_value_codes(self, X):
        """
        Look for data that might have been coded as "missing" by writing it as a
        long string of 9's.

        :effect self._potential_missing_value_codes:
            Populated with a list of potential missing value codes.
        """
        self._potential_missing_value_codes = []
        if self._number_unique < self._number_observed:
            # Get the 100 most frequent values as a collection of integers.
            frequency_distribution = X.value_counts().sort_values(
                ascending=False)

            # Consider at most the top 100 most frequent values, and only
            # consider values that occur more than 10 times.
            if len(frequency_distribution) > 100:
                frequency_distribution = frequency_distribution.iloc[:100]
            frequency_distribution = frequency_distribution[
                frequency_distribution > 10]

            candidates = np.array([is_all_nines(val)
                                   for val in frequency_distribution.index])
            if np.any(candidates):
                self._potential_missing_value_codes = (
                    frequency_distribution.index[candidates].tolist()
                )

    def __repr__(self):
        ans = f"""
        Nobs: {self._number_observed}
        Nmis: {self._number_missing}
        Nunique: {self._number_unique}

        Mean: {self._mean :.4f}
          SD: {self._sd :.4f}

        Quantiles:
{self._quantiles}
        """
        return ans


class CategoricalSummary(UnivariateSummary):
    """
    Standard univariate summaries of a categorical variable.
    """

    def __init__(self, x, max_levels: int = 10,
                 other_name: str = "[Missing]"):
        """
        Args:
          x: The categorical variable to summarize.  This will be coerced to a
            categorical pd.Series.  If None the object is created empty, with
            the expection that it will be immediately filled by other methods,
            e.g. through deserialization.

          max_levels: The maximum number of non-missing levels to consider.
            The most frequent levels will be captured. If max_levels = None,
            all levels will be captured.

          other_name: Name to use for levels that are 'collapsed' due to high
            cardinality.  The chosen name should be one that is highly unlikely
            to appear in the raw data.  The default is chosen to appear near the
            end of alphabetically sorted lists.
        """
        if x is not None:
            self._summarize(x, max_levels, other_name)

    @classmethod
    def from_counts(cls,
                    counts,
                    max_levels=None,
                    other_name: str = "[Missing]"):
        """
        Generate a CategoricalSummary from a pd.Series of category counts, or a
        FrequencyDistribution.

        Args:
          counts: Either a pd.Series or a FrequencyDistribution describing the
            variable.
          max_levels: The maximum number of non-missing levels to consider.  The
            most frequent levels will be captured. If max_levels = None, all
            levels will be captured.
          other_name: Name to use for levels that are 'collapsed' due to high
            cardinality.  The chosen name should be one that is highly unlikely
            to appear in the raw data.  The default is chosen to appear near the
            end of alphabetically sorted lists.
        """
        from .frequency_distribution import FrequencyDistribution

        obj = cls(None)
        if isinstance(counts, pd.Series):
            obj._frequency_distribution = (
                FrequencyDistribution.from_counts(counts)
            )
        elif isinstance(counts, FrequencyDistribution):
            obj._frequency_distribution = counts
        else:
            raise Exception(
                "counts should be a FrequencyDistribution or a pd.Series.")

        obj._promote_frequency_distribution(
            max_levels=max_levels, other_name=other_name)
        return obj

    @property
    def sample_size(self):
        """
        The total length of the variable being summarized.
        """
        return self._number_missing + self._number_observed

    @property
    def number_missing(self):
        return self._number_missing

    @property
    def number_observed(self):
        return self._number_observed

    @property
    def number_unique(self):
        """
        The number of distinct values.  Missing values count as a separate
        category.  If the summary was computed by merging two distributed
        summaries then this number may be a lower bound.
        """
        return self._number_unique

    @property
    def is_binary(self):
        """
        A variable is binary if it has 2 categories.  If missing values are
        present they are counted as a category, so [1, 0, nan] is not binary.
        """
        return len(self._frequency_distribution) == 2

    @property
    def has_missing_data(self):
        return self.number_missing > 0

    @property
    def is_high_cardinality(self):
        return len(self.levels(True)) >= self.cardinality_limit

    @property
    def cardinality_limit(self):
        limit = max(5, int(self.sample_size ** (10.0 / 37)))
        return int(limit)

    @classmethod
    def default_other_category_name(cls):
        return "[Other]"

    @property
    def other_category_name(self):
        """
        The name of the category into which high cardinality levels were
        collapsed. If no collapsing was done this is the empty string.
        """
        return self.frequency_distribution.other_category_name

    def levels(self,
               include_missing: bool = False,
               missing_code=np.nan,
               omit_zero_frequency: bool = True):
        """
        Returns the levels for encoding the categorical variable.

        Args:
          include_missing: Include missing values as a separate level, if any
            missing values are present in the summarized data.
          missing_code: Value used to represent missing data in the returned
            list.
          omit_zero_frequency: Should levels that are known to exist, but which
            have zero frequency in the data, be omitted from the output.

        Returns:
            List of the most frequent levels, up to max_levels.
        """
        fd = self.frequency_distribution
        if omit_zero_frequency:
            fd = fd[fd > 0]
        ans = fd.index.tolist()
        if include_missing and self.frequency_distribution.has_missing_data:
            ans.append(missing_code)
        return ans

    @property
    def frequency_distribution(self):
        """
        The distribution of categories, expressed as counts.
        """
        return self._frequency_distribution

    @property
    def relative_frequency_distribution(self):
        """
        The distribution of categories, expressed as probabilities.
        """
        return self.frequency_distribution / self.sample_size

    def __repr__(self):
        return f"""


    Nobs:       {self._number_observed}
    Nmis:       {self._number_missing}
    Nunique:    {self._number_unique}

Frequency Distribution:
{pad(self._frequency_distribution)}
            """

    def _promote_frequency_distribution(self, max_levels, other_name):
        """
        Generate the other summary atributes based on
        self._frequency_distribution.

        Args:
          max_levels: The maximum number of non-missing levels to consider.
            The most frequent levels will be captured. If max_levels = None,
            all levels will be captured.
          other_name: Name to use for levels that are 'collapsed' due to high
            cardinality.  The chosen name should be one that is highly unlikely
            to appear in the raw data.  The default is chosen to appear near
            the end of alphabetically sorted lists.

        Effects:
          The following fields are populated:
          - self._number_missing:
          - self._number_observed:
          - self._number_unique:

          The field self._frequency_distribution is potentially collapsed if
          levels exceed max_levels.
        """

        if self._frequency_distribution.has_missing_data:
            self._number_missing = self._frequency_distribution[np.nan]
        else:
            self._number_missing = 0
        sample_size = np.sum(self._frequency_distribution)
        self._number_observed = sample_size - self._number_missing

        # The number of unique values in the original data, prior to
        # collapsing.  The "missing value" level, if present, is included in
        # the count.
        self._number_unique = len(self._frequency_distribution)
        self._frequency_distribution.collapse(max_levels, other_name)

    def _summarize(self,
                   x,
                   max_levels: int = 10,
                   other_name: str = "[Other]"):
        """
        Args:
          x: The categorical variable to summarize.  This will be coerced to a
            categorical pd.Series.
          max_levels: The maximum number of non-missing levels to consider.  The
            most frequent levels will be captured. If max_levels = None, all
            levels will be captured.
          other_name: Name to use for levels that are 'collapsed' due to high
            cardinality.  The chosen name should be one that is highly unlikely
            to appear in the raw data.  The default is chosen to appear near the
            end of alphabetically sorted lists.

        Effects:
          self._frequency_distribution is created.
        """
        from .frequency_distribution import FrequencyDistribution
        try:
            # Calling with x as None should only be done by another constructor,
            # when the aim is to fill in class slots manually.
            if x is None:
                return

            if (
                    isinstance(x, np.ndarray)
                    and len(x.shape) == 2
                    and x.shape[1] == 1
            ):
                x = x.ravel()

            self._frequency_distribution = FrequencyDistribution(x)
            self._promote_frequency_distribution(
                max_levels=max_levels, other_name=other_name)

        except Exception as e:
            message = "Could not summarize a categorical variable with "
            message += "initial values"
            message += f" {x.head()} and dtype {x.dtype}.\n"
            message += "Original error message: " + str(e)
            raise Exception(message)


class DateTimeSummary(UnivariateSummary):
    """
    A summary of a DateTime variable.

    Dates are often associated with sequential observations, and often appear
    in order:  Sales by Date.

    Dates can also be attributes of events that might not be sorted in the
    data: Order Date.
    """

    def __init__(self, x):
        # The only time the constructor should be called with x as None is for
        # special purposes like JSON deserialization.
        if x is None:
            return
        self._summarize(x)

    def _summarize(self, x):
        from .frequency_distribution import FrequencyDistribution
        try:
            x = pd.Series(x, dtype="datetime64[ns]")
            self._sample_size = len(x)
            self._number_observed = x.count()
            self._number_missing = self._sample_size - self._number_observed
            unique_timestamps = pd.Series(x.unique()).sort_values()
            self._number_unique = len(unique_timestamps)
            self._min = np.min(unique_timestamps)
            self._max = np.max(unique_timestamps)
            self._granularity = granularity(x)

            self._weekday_counts = FrequencyDistribution.from_counts(
                pd.Series(weekday(x)).value_counts()[day_names],
                self._number_missing if self._number_missing > 0 else None,
                categories=day_names,
            )

            self._month_counts = FrequencyDistribution.from_counts(
                pd.Series(month(x)).value_counts()[month_names],
                self._number_missing if self._number_missing > 0 else None,
                categories=month_names,
            )

            self._compute_intensity(x)
            self._regular = self._check_regularity(x)

        except Exception as e:
            message = "Could not summarize a datetime variable with initial "
            message += f"values {x.head()} "
            message += f"and dtype {x.dtype}.\n"
            message += "Original error message: " + str(e)
            raise Exception(message)

    def __repr__(self):
        ans = f"""
    Nobs:       {self.number_observed}
    Nmis:       {self.number_missing}
    Nunique:    {self.number_unique}
    Min:        {self.min}
    Max:        {self.max}
    Duration:   {self.max - self.min}
    Granularity:  {self._granularity}
    Regular:    {self.regular}
        """
        if self._intensity is not None:
            ans += f"""

Intensity:
{pad(self._intensity, 4)}
            """
        ans += f"""

Day of week seasonality:
{pad(self._weekday_counts, 4)}

Monthly seasonality:
{pad(self._month_counts, 4)}
        """
        return ans

    @property
    def sample_size(self):
        return self._sample_size

    @property
    def number_missing(self):
        return self._number_missing

    @property
    def number_observed(self):
        return self._sample_size - self._number_missing

    @property
    def number_unique(self):
        return self._number_unique

    @property
    def min(self):
        """
        The earliest observed time point.
        """
        return self._min

    @property
    def max(self):
        """
        The latest observed time point.
        """
        return self._max

    @property
    def intensity(self):
        """
        A frequency distribution of the number of observations per time period.
        """
        return self._intensity

    @property
    def weekday_counts(self):
        """
        A frequency distribution showing the day-of-week pattern in the data.
        The first element corresponds to Monday.
        """
        return self._weekday_counts

    def summarize_weekdays(self):
        """
        Return a CategoricalSummary object describing the weekday effect.
        """
        return CategoricalSummary.from_counts(self.weekday_counts)

    @property
    def month_counts(self):
        """
        A frequency distribution showing the monthly annual seasonal pattern.
        """
        return self._month_counts

    def summarize_monthly(self):
        """
        Return a CategoricalSummary describing the monthly annual seasonal
        pattern.
        """
        return CategoricalSummary.from_counts(self.month_counts)

    @property
    def duration(self):
        """
        The total amount of time covered by the variable.
        """
        return self._max - self._min

    @property
    def granularity(self):
        """
        The smallest time delta.  This is a pd.Timedelta.
        """
        return self._granularity

    @property
    def granularity_bucket(self):
        return granularity_bucket(self._granularity)

    @property
    def regular(self):
        """
        A series is regular if all its time difference are the same sign, and
        almost all are the same value.
        """
        return getattr(self, "_regular", False)

    def group_timestamps(self, dates):
        """
        Return a set of discrete identifiers for the time group to which each
        entry of 'dates' belongs.  Longer time windows are chopped into coarser
        buckets.  The return values are either based on years, months, weeks, or
        days depending on the time window covered by this summary.

        Args:
          dates: A pd.Series of dtype datetime64[ns], or compatible.

        Returns:
          A pd.Series of string labels.  This is a coarsening of 'dates'
          representing the date group to which each date belongs.  The return
          value must maintain equivalence between lexicographic and datetime
          sortability.
        """
        if self.duration.days > (365 * 5):
            return pd.Series(dates.dt.year).astype(str)
        elif self.duration.days > 365:
            years = dates.dt.year.astype(str).values
            months = dates.dt.month.astype(str).str.zfill(2).values
            return pd.Series(years + "-" + months)
        elif self.duration.days > 30:
            # For weekly data, the axis label is the day that starts the week.
            #
            # This is the proper method in pandas0.23, but it is deprecated in
            # pandas 1.0 in favor of dates.dt.isocalendar().week.
            #
            # .normalize() sets the time to 00.00.00, so we're just dealing with
            # days.
            week = pd.Series(dates, dtype="datetime64[ns]").dt.normalize()
            week -= pd.to_timedelta(week.dt.weekday, unit="D")
            years = week.dt.year.astype(str).values
            months = week.dt.month.astype(str).str.zfill(2).values
            days = week.dt.day.astype(str).str.zfill(2).values
            return pd.Series(years + "-" + months + "-" + days)
        elif self.duration.days > 1:
            days = pd.Series(dates, dtype="datetime64[ns]").dt.normalize()
            years = days.dt.year.astype(str).values
            months = days.dt.month.astype(str).str.zfill(2).values
            days2 = days.dt.day.astype(str).str.zfill(2).values
            return pd.Series(years + "-" + months + "-" + days2)
        elif self.duration.seconds > 0:
            hours = dates.dt.hour.astype(str).str.zfill(2).values
            minutes = dates.dt.minute.astype(str).str.zfill(2).values
            seconds = dates.dt.second.astype(str).str.zfill(2).values
            times = pd.Series(hours + ":" + minutes + ":" + seconds)
            if self.min.day == self.max.day:
                return times
            else:
                years = dates.dt.year.astype(str).values
                months = dates.dt.month.astype(str).str.zfill(2).values
                days = dates.dt.day.astype(str).str.zfill(2).values
                datestamp = pd.Series(years + "-" + months + "-" + days)
                return datestamp + " " + times

        else:
            return dates.astype(str)

    def _to_categorical_summary(self, frequency_distribution):
        """
        Promote a frequency_distribution to a CategoricalSummary object.
        """
        ans = CategoricalSummary(None)
        ans._number_observed = self.number_observed
        ans._number_missing = self.number_missing
        ans._number_unique = self.number_unique
        ans._frequency_distribution = frequency_distribution
        ans._other_category_name = ""
        return ans

    def _compute_intensity(self, dates):
        """
        Returns the number of times a data occurred in each grouping period.
        The grouping period used is a function of the time window covered by
        the date range.

        Args:
          dates: pd.Series of dtype datetime64[ns], or equivalent type.

        Effects:
          self._intensity is assigned a pd.Series, indexed by the time groups,
            containing the count associated with each group.
        """
        self._intensity = self.group_timestamps(
            dates).value_counts().sort_index()

    def _check_regularity(self, timestamps):
        """
        A group of time stamps is regular if it is ordered and 99% or more of
        the time differences are the same value.
        """
        if self._number_unique < 0.9 * self._number_observed:
            return False
        if self.number_missing / self.sample_size >= 0.5:
            return False
        timestamps = pd.Series(timestamps, dtype="datetime64[ns]")
        dt = timestamps.diff()
        dt_seconds = dt / np.timedelta64(1, "s")
        if np.any(dt_seconds < 0):
            dt_seconds = -1 * dt_seconds
            if np.any(dt_seconds < 0):
                return False
        summary = CategoricalSummary(dt_seconds)
        sorted_dist = summary.relative_frequency_distribution.sort_values(
            ascending=False)
        # sorted_dist[0] is fraction of the time the most commonly occurring
        # 'dt' occurs.  sorted_dist[0] is the value of dt that occurs most
        # often.  If the value is "{Other}" then there are lots of distinct
        # dt's.  If it is a very frequent numeric value then we've got a
        # regular series.
        #
        # Setting the threshold to 5/7 allows for "weekday" data to be
        # considered regular if weekends are always missing.
        if sorted_dist[0] >= 4.9 / 7 and isinstance(
                sorted_dist.index[0], float):
            return True

        # Special handling for monthly data.  If all time deltas are at least
        # 28 days and most are less than 31 then assume this is regular monthly
        # data.
        dt_days = dt / pd.Timedelta(1, "day")
        dt_days = dt_days.dropna()
        dt_days = dt_days[dt_days > 0]
        if np.all(dt_days >= 28) and np.mean(dt_days <= 31) >= 0.8:
            return True

        # Special handling for quarterly data.  If all time deltas are at least
        # 90 days, and most are less than 93 then assume quarterly data.
        if np.all(dt_days >= 90) and np.mean(dt_days < 93) >= 0.8:
            return True

        return False

    @staticmethod
    def _deduce_intensity_granularity(intensity: pd.Series):
        """
        Args:
          intensity: An intensity function.

        Returns: A string describing which of the level of granularity in the
           intensity function.

           The index of an intensity function is a vector of strings with one
           of the following forms:

            1) 2020                (Yearly)
            2) 2020-04             (Monthly)
            3) 2020-04-03          (Weekly)
            4) 2020-04-03          (Daily)
            5) 2020-04-03 16:08:24 (SubDay-DateTime)
            6) 16:08:24            (Subday-TimeOnly)
        """
        date_chunks = intensity.index[0].split("-")
        if len(date_chunks) == 1:
            chunk = date_chunks[0]
            return "Yearly" if len(chunk) == 4 else "SubDay-TimeOnly"
        elif len(date_chunks) == 2:
            # Dates are year-month
            return "Monthly"
        elif len(date_chunks) == 3:
            day_chunk = date_chunks[2]
            if len(day_chunk) > 2:
                return "SubDay-DateTime"
            else:
                unique_timestamps = pd.to_datetime(pd.Series(intensity.index))
                dt = unique_timestamps[1] - unique_timestamps[0]
                return "Weekly" if dt.days == 7 else "Daily"
