import pandas as pd
import numpy as np


class Dframe(pd.DataFrame):
    """A data frame that indexes like R.  self[] takes two entries, rows and
    columns.

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


def ls(*args, hide_underscore=True):
    """ List the contents of one or more objects.
    """
    if len([*args]) == 0:
        pretty(sorted(globals()), hide_underscore=hide_underscore)
    else:
        for i in range(len(args)):
            pretty(dir(args[i]), hide_underscore=hide_underscore)
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


def range(x):
    """
    Return the smallest and largest entries in x.
    """
    return np.quantile(x, q=[0, 1])
