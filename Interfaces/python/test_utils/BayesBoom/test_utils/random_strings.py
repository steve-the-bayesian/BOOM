import numpy as np
import string


def random_strings(num_strings, length, ensure_unique: bool = True):
    """
    Args:
      num_strings: The length of the returned array.
      length:  The length of each string in the array.
      ensure_unique: If True then each string is the returned array will be
        unique.  If False then duplicate entries are possible.

    Returns:
      An array of lower case ASCII strings.
    """
    if ensure_unique and (num_strings > 26**length):
        raise Exception("Too many strings requested")
    letters = np.random.choice(list(string.ascii_lowercase),
                               (num_strings, length))
    ans = np.apply_along_axis(lambda x: "".join(x),
                              1,
                              letters)

    while len(np.unique(ans)) < num_strings:
        levels, counts = np.unique(ans, return_counts=True)
        duplicates = levels[counts > 1]
        dup_counts = counts[counts > 1]
        ans[ans == duplicates[0]] = random_strings(dup_counts[0], length)
    return ans
