import pandas as pd
import numpy as np
from inspect import isfunction, getsource
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


def pretty(list_of_strings, width=80, hide_underscore=True):
    '''Print a list of strings in formatted columns.

    Args:
      list_of_strings: The list of strings to print.
      width:  The width of the screen.
      hide_underscore: If True then elements in list_of_strings that begin or
        end with underscores will not be printed.

    Returns:
      None

    Effects:
      The strings are printed to the screen.
    '''
    if hide_underscore:
        private = [(x[0] == '_') | (x[-1] == '_') for x in list_of_strings]
        to_print = [x for x, hide in zip(
            list_of_strings, private) if hide is False]
    else:
        to_print = list_of_strings

    if len(to_print) == 0:
        return
    max_len = np.max([len(x) for x in to_print])
    entry_width = max_len + 2

    line = ''
    for entry in to_print:
        # This would be a good place to use string formatting, but I had
        # trouble getting it to work.
        buffer = ' ' * (entry_width - len(entry))
        padded = entry + buffer
        if len(line) + entry_width <= width:
            line += padded
        else:
            print(line)
            line = padded
    print(line)


def print_timestamp(iteration_number, ping):
    if ping <= 0:
        return
    if iteration_number % ping == 0:
        timestamp = time.asctime()
        sep = ' =-=-=-=-=-=-=-=-=-=-= '
        print(sep + timestamp + f" Iteration {iteration_number} " + sep)


def ls(*args, hide_underscore=True):
    """
    List the contents of one or more objects.  If passed a function then print
    the body of the function.
    """

    if len([*args]) == 0:
        print("If you're trying to list the global namespace type "
              "'pretty(dir())'.")
        print("(The interactive namespace is not available to modules.)")
#        pretty(sorted(globals()), hide_underscore=hide_underscore)
    elif len([*args]) == 1 and isfunction(args[0]):
        print(getsource(args[0]))
    else:
        for arg in args:
            pretty(dir(arg), hide_underscore=hide_underscore)
            print("\n")


# Need the formula language so we can ask for conditional distributions.  This
# function is better than pd.value_counts because it handles numpy and list
# data too.
def table(*args):
    """
    Compute a frequency table of one or more categorial variables.
    """
    if len(args) == 1:
        if isinstance(args[0], pd.DataFrame):
            return args[0].crosstab(margins=True)
        else:
            x = pd.Series(args[0])
            return x.value_counts()
    else:
        x = pd.DataFrame(*args)
        return x.crosstab(margins=True)


def data_range(x):
    """
    Return the smallest and largest entries in x.  The name distinguishes this
    function from the python built-in 'range'.
    """
    return np.quantile(x, q=[0, 1])


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
    Paste one or more vector-like objects to gether into a vector of strings.

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
