import numpy as np
import string


def random_strings(n, string_length=5):
    """
    Return 'n' randomly generated strings of length 'string_length' formed by
    randomly selecting lower case ascii letters with replacement.
    """
    letters = np.random.choice(list(string.ascii_lowercase),
                               size=(string_length, n),
                               replace=True)
    ans = np.apply_along_axis(lambda x: "".join(x),
                              0,
                              letters)
    return ans
