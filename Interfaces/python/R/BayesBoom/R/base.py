import pandas as pd
import numpy as np
import time
from numbers import Number


class omit:
    """
    For use in indexing. my_data[omit(bad_integers), :] omits the bad rows.
    """


class data_frame(pd.DataFrame):
    """A data frame that indexes like R.  self[] takes two entries, rows and
    columns.

    TODO: flesh out the indexer, and check that all the pandas methods still
    work.

    """

    def __init__(self, *args, **kwargs):
        """The constructor is identical to pandas constructor.
        """
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        return super().__getitem__(name)

    def __getitem__(self, arg):
        if not isinstance(arg, tuple) and not len(arg) == 2:
            raise Exception("Rows and columns must be supplied.")

        rows = arg[0]
        cols = arg[1]
        print("rows =  ", rows)
        print("cols =  ", cols)
        # TODO actually do the indexing.
        # In most cases indexing by columns first is the right thing to do.

        # If slice and not None, None, None

    @property
    def nrow(self):
        return self.shape[0]

    @property
    def ncol(self):
        return self.shape[1]


def print_timestamp(iteration_number=None, ping=None):
    sep = ' =-=-=-=-=-=-=-=-=-=-= '
    if iteration_number is None:
        print(sep, time.asctime(), sep)
    elif (ping is not None) and (ping > 0) and (iteration_number % ping == 0):
        timestamp = time.asctime()
        print(sep + timestamp + f" Iteration {iteration_number} " + sep)


def print_time_interval(seconds: float, print_output=True):
    """
    Print a time interval in English, using days, hours, minutes and seconds.
    """
    seconds_in_minute = 60
    seconds_in_hour = seconds_in_minute * 60
    seconds_in_day = seconds_in_hour * 24
    remaining_seconds = seconds

    days = int(remaining_seconds / seconds_in_day)
    remaining_seconds = remaining_seconds % seconds_in_day

    hours = int(remaining_seconds / seconds_in_hour)
    remaining_seconds = remaining_seconds % seconds_in_hour

    minutes = int(remaining_seconds / seconds_in_minute)
    remaining_seconds = remaining_seconds % seconds_in_minute

    integer_seconds = int(remaining_seconds)
    fractional_seconds = remaining_seconds % 1

    ans = ""
    day_unit = "day" if days == 0 else "days"
    hour_unit = "hour" if hours == 1 else "hours"
    minute_unit = "minute" if minutes == 1 else "minutes"

    if days > 0:
        ans += str(days) + " " + day_unit
    if hours > 0 or days > 0:
        ans += " " if len(ans) > 0 else ""
        ans += str(hours) + " " + hour_unit
    if days > 0 or hours > 0 or minutes > 0:
        ans += " " if len(ans) > 0 else ""
        ans += str(minutes) + " " + minute_unit
    ans += " " if len(ans) > 0 else ""
    if fractional_seconds == seconds:
        ans += str(fractional_seconds)
    else:
        ans += f"{integer_seconds + fractional_seconds:.3f}"
        ans += " seconds"
    if print_output:
        print(ans)
    return ans


# Need the formula language so we can ask for conditional distributions.  This
# function is better than pd.value_counts because it handles numpy and list
# data too.
def table(*args):
    """
    Compute a frequency table of one or more categorical variables.
    """
    if len(args) == 1:
        if isinstance(args[0], pd.DataFrame):
            return args[0].crosstab(margins=True)
        elif isinstance(args[0], np.ndarray):
            values, counts = np.unique(args[0], return_counts=True)
            ans = pd.Series(counts, index=values)
            return ans.sort_index()
        else:
            x = pd.Series(args[0])
            return x.value_counts()
    elif len(args) == 2:
        x = args[0]
        y = args[1]
        return _fast_crosstab(x, y)
    else:
        # the return value is a pd.DataFrame with args[0] as the index and the
        # cartesian product of the other args as columns.  This isn't the most
        # useful form, but improving it is a TODO.
        return pd.crosstab(args[0], list(args)[1:], margins=True)


def _fast_crosstab(x1, x2, xname="X", yname="Y", dropna=False):
    """
    Args:
       x1, x2:
          pd.Categorical data, or objects convertable to such.
    """
    x1 = pd.Categorical(x1)
    x2 = pd.Categorical(x2)

    n = len(x1)
    if len(x2) != n:
        raise Exception("x1 and x2 must have the same length")

    xindex = x1.categories.tolist()
    yindex = x2.categories.tolist()

    nlev = [len(xindex), len(yindex)]
    has_na = [False, False]
    if dropna:
        x1 = x1.dropna()
        x2 = x2.dropna()

    else:
        if x1.codes.min() == -1:
            nlev[0] += 1
            has_na[0] = True
        if x2.codes.min() == -1:
            nlev[1] += 1
            has_na[1] = True

    X1 = np.zeros((n, nlev[0]))
    X1[np.arange(n), x1.codes] = 1

    X2 = np.zeros((n, nlev[1]))
    X2[np.arange(n), x2.codes] = 1

    xtabs = X1.T @ X2
    xindex = x1.categories.tolist()
    yindex = x2.categories.tolist()
    if has_na[0]:
        xindex.append("NA")
    if has_na[1]:
        yindex.append("NA")

    ans = pd.DataFrame(xtabs, index=xindex, columns=yindex, dtype=int)
    ans.index.name = xname
    ans.columns.name = yname
    return ans


def order(input_sequence):
    """
    Given an input sequence, return a vector of integers that will put the
    sequence in order.

    """
    return np.argsort(input_sequence)


def invert_order(ordr):
    """
    Put entries that have been sorted by a call to 'order' back in their
    original order.

    Args:
      ord: A permutation of the numbers 0, ... n.

    Returns:
      ans: A permutation of the numbers 0..n such that ans[ord] = 0...n.
    """
    n = len(ordr)
    return pd.Series(range(n), index=ordr).sort_index().values


def data_range(x):
    """
    Return the smallest and largest entries in x.  The name distinguishes this
    function from the python built-in 'range'.
    """
    return np.quantile(x, q=[0, 1])


def var(x):
    """
    Compute the variance of the input x.  If x is a vector then return the
    scalar valued variance.  If x is a matrix return the variance matrix,
    assuming each row of x is an observation.
    """
    if isinstance(x, Number):
        return 0
    elif len(x.shape) == 1:
        return np.var(x, ddof=1)
    else:
        return np.cov(x, rowvar=False, ddof=1)


def corr(*args):
    """
    Compute the correlation among one or more objects.  If a single matrix or
    data frame is passed, or if a collection of vectors is passed, then return
    the correlation matrix.  If a pair of vectors is passed, return the number.

    This function corrects the stupid default in numpy which assumes variables
    are rows rather than columns.
    """
    if len(args) == 1:
        x = args[0]
    else:
        x = np.stack(args, axis=1)
        if x.shape[1] == 2:
            return np.corrcoef(x, rowvar=False)[0, 1]
    return np.corrcoef(x, rowvar=False)


def first_true(boolean_array):
    """
    Returns the index of the first True element in the array-like boolean_array.
    Returns None if no True values are found.
    """
    return next((i for i, v in enumerate(boolean_array) if v), None)


def which(boolean_array):
    """
    Return the integer indices at which 'boolean_array' is True.
    """
    n = len(boolean_array)
    indices = np.arange(n)
    return indices[boolean_array.astype(bool)]


def recycle(x, output_len):
    x = list(x)
    nchoices = len(x)
    if nchoices >= output_len:
        return x[:output_len]
    ans = x
    len_ans = nchoices
    while len_ans + nchoices < output_len:
        ans = ans + x
        len_ans += nchoices
    num_remaining = output_len - len_ans
    return ans + x[:num_remaining]


def unique_match(value, legal_value_list):
    """
    If 'value' uniquely matches only one value in legal_value_list, then return
    the corresponding value in legal_value_list.  If not then return None.
    """

    matches = np.array([x.startswith(value) for x in legal_value_list])
    if matches.sum() != 1:
        return None
    else:
        return legal_value_list[first_true(matches)]


def _reduce_concat(x, sep=""):
    import functools
    return functools.reduce(lambda x, y: str(x) + sep + str(y), x)


def _deduce_type(*lists):
    """
    Return an integer indicating the "most general" type in the variable length
    argument list *lists.  Return type 2 is pd.Series, return type 1 is
    np.array, and return type 0 is list.

    Args:
      *lists: a sequence of vector-like objects.
    """
    type_code = 0
    for obj in lists:
        if isinstance(obj, pd.Series):
            type_code = 2
        elif isinstance(obj, np.ndarray):
            type_code = max(type_code, 1)
    return type_code


def paste(*lists, sep=" ", collapse=None):
    """
    Paste one or more vector-like objects together into a vector of strings.

    Args:
      *lists:  A sequence of vector-like objects.

    Returns:
      A vector-like object containing the concatenated string representations
      of the arguments.  If any of the arguments are a pd.Series then the
      return type is also pd.Series.  Otherwise if any arguments are numpy
      arrays the return is a numpy array.  Otherwise the return type is a list.
    """
    list_of_inputs = [*lists]
    max_length = np.max([len(x) for x in list_of_inputs])
    for i in range(len(list_of_inputs)):
        if isinstance(list_of_inputs[i], str):
            string_value = list_of_inputs[i]
            list_of_inputs[i] = [string_value] * max_length
        elif isinstance(list_of_inputs[i], Number):
            value = list_of_inputs[i]
            list_of_inputs[i] = [value] * max_length

    parallel_data = pd.DataFrame(list_of_inputs).T
    result = parallel_data.apply(_reduce_concat, sep=sep, axis=1).astype(
        str).values.tolist()
    if collapse is not None:
        return _reduce_concat(result, sep=collapse)
    else:
        return result


def paste0(*lists, sep="", collapse=None):
    return paste(*lists, sep=sep, collapse=collapse)


def remove_common_prefix(strings):
    if len(strings) == 0:
        return strings
    current = [x for x in strings]
    while True:
        try:
            initial_element = {x[0] for x in current}
        except IndexError:
            return current
        if len(initial_element) > 1:
            return current
        current = [x[1:] for x in current]


def remove_common_suffix(strings):
    """
    If all strings in a collection end in the same character, that character is
    a 'common suffix."  This function reomves common suffixes from collections
    of strings.

    Args:
      strings:  A list, array, or other collection of strings.

    Returns:
      shortened_strings:  A list of string with common suffixes removed.

    Examples:
      remove_common_suffix([])
      []

      remove_common_suffix(["foo", "bar", "baz"])
      ["foo", "bar", "baz"]

      remove_common_suffix(["fooz", "barz", "baz"])
      ["foo", "bar", "ba"]

      remove_common_suffix(["zz", "z"])
      ["z", ""]

    """
    if len(strings) == 0:
        return strings
    current = [x for x in strings]
    while True:
        try:
            final_element = {x[-1] for x in current}
        except IndexError:
            return current
        if len(final_element) > 1:
            return current
        current = [x[:-1] for x in current]
