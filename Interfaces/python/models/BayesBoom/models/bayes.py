"""
Python wrappers around BayesBoom C++ model objects.

Each wrapper holds the parameters of its corresponding C++ model in plain
Python/NumPy form and provides a .boom() method that returns the live C++
object.  This shields callers from C++ type-finickiness (e.g. passing a
Python int where boom expects a double, or a regular numpy array where boom
expects a boom.Vector).

Plotting helpers (plot_components, etc.) lazily import from BayesBoom.R when
called, so BayesBoom.R is an optional dependency for that functionality.
DataBuilder helpers used by MixtureComponent subclasses are likewise lazily
imported from BayesBoom.R.
"""

import numpy as np
import pandas as pd
import copy

import matplotlib.pyplot as plt

from .boom_utils import (
    to_boom_vector,
    to_boom_matrix,
    to_boom_spd,
    unique_match,
)


# ---------------------------------------------------------------------------
# Scalar priors / distributions
# ---------------------------------------------------------------------------

