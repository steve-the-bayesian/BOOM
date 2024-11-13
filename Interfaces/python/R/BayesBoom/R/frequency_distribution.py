import pandas as pd
import numpy as np


class FrequencyDistribution:
    """
    The frequency distribution of a categorical variable.  Missing values are
    kept as a separate count.
    """

    def __init__(self, variable, categories=None):
        """
        Args:
          variable: The categorical variable to be summarized by a frequency
            table.
          categories: The set of categories to be tabulated.  If None then all
            categories will be counted.  If 'categories' is supplied then any in
            the variable not in 'categories' gets counted as 'other'.
        """
        if variable is None:
            return

        if isinstance(variable, np.ndarray):
            variable = pd.Series(variable, dtype="category")
        nans = variable.isna()

        if not nans.all():
            counts = variable.value_counts(dropna=True)
        else:
            counts = pd.Series(dtype=int)

        self._other_category_name = "[Other]"
        self._non_nan = counts
        self._nan_counts = nans.sum()

    @classmethod
    def from_counts(cls, non_nan_counts, nan_count: int = None,
                    categories=None):
        """
        Construct a FrequencyDistribution from a set of counts and categories.

        Args:
          non_nan_counts:
            A pd.Series containing counts of the non_nan category values.

          nan_count: The number of observations in the nan category.  If 0 is
            entered then nan will be present with a count of zero.  If None is
            entered then nan will not be a category.

          categories: Array-like list of category labels, with the same length
            as 'non_nan'.  If 'None' then the index of non_nan will be used,
            which is the usual, expected case.
        """
        obj = cls(None)
        obj._non_nan_counts = non_nan_counts
        obj._nan_counts = nan_count
        obj._other_category_name = "[Other]"
        return obj

    @property
    def levels(self):
        ans = self._non_nan.index.tolist()
        return ans

    @property
    def dtype(self):
        if len(self._non_nan) > 0:
            return self._non_nan.dtype
        else:
            return np.dtype(int)

    @property
    def other_category_name(self):
        return self._other_category_name

    def __repr__(self):
        ans = str(self._non_nan)
        if ans._nan_counts > 0:
            ans += f"nan: {self._nan_counts}"
        return ans
