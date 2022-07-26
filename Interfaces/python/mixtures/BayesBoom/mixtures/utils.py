import numpy as np
import pandas as pd

# A collection of free utility functions helpful for mixture modeling.


def normalize_logprob(log_probs):
    """
    Convert a matrix of unnormalized log probabilities into normalized
    probabilities.

    Args:
      log_probs: A matrix where each row is a set of log probabilities to be
        exponentiated and normalized so the exponentiated row sums to one.
    """
    if isinstance(log_probs, (pd.DataFrame, pd.Series)):
        log_probs = log_probs.values

    row_max = log_probs.max(axis=1).reshape((-1, 1))
    probs = np.exp(log_probs - row_max)
    row_totals = probs.sum(axis=1).reshape((-1, 1))
    return probs / row_totals
