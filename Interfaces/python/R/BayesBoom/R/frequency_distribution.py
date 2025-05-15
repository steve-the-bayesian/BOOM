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

        if not categories:
            variable = pd.Series(variable, dtype="category")
        else:
            variable = pd.Series(variable, dtype=pd.CategoricalDtype(categories))
                
        nans = variable.isna()

        if not nans.all():
            counts = variable.value_counts(dropna=True)
        else:
            # If the input sequence was all nan's then counts is an empty series.
            counts = pd.Series(dtype=int)

        self._other_category_name = "[Other]"

        # Put the counts in the order of the categories, if one was given.
        self._non_nan = counts[variable.cat.categories]
        self._nan_counts = int(nans.sum())

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
        obj._nan_counts = int(nan_count)
        obj._other_category_name = "[Other]"
        return obj

    @property
    def counts(self):
        """
        The count of missing levels other than NaN's
        """
        return self._non_nan

    @property
    def nan_count(self):
        return self._nan_counts
    
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

    @property
    def sample_size(self):
        return int(self._non_nan.sum() + self._nan_counts)
    
    def __repr__(self):
        ans = str(self._non_nan)
        if self._nan_counts > 0:
            ans += f"nan: {self._nan_counts}"
        return ans

