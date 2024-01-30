import numpy as np


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
