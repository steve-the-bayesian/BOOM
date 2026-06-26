"""
Low-level utilities for converting Python/NumPy objects to their BayesBoom
C++ counterparts.  These functions are intentionally kept self-contained so
that BayesBoom.models does not require BayesBoom.R at import time.
"""

import numpy as np
from numbers import Number


def to_boom_vector(v):
    """
    Convert a vector-like object to boom.Vector.  Accepts numpy arrays of any
    numeric dtype, pandas Series, scalars, or an existing boom.Vector.
    """
    import BayesBoom.boom as boom
    if isinstance(v, boom.Vector):
        return v
    if hasattr(v, "values"):
        return boom.Vector(v.to_numpy(dtype="float").copy())
    if isinstance(v, Number):
        return boom.Vector(np.array([v], dtype="float"))
    return boom.Vector(np.array(v, dtype="float"))


def to_boom_matrix(m):
    """
    Convert a matrix-like object to boom.Matrix.  Accepts numpy arrays,
    pandas DataFrames containing numeric data, scalars, or an existing
    boom.Matrix.
    """
    import BayesBoom.boom as boom
    if isinstance(m, boom.Matrix):
        return m
    if hasattr(m, "values") and hasattr(m, "dtypes"):
        return boom.Matrix(m.values.astype("float").copy())
    if isinstance(m, Number):
        return boom.Matrix(np.full((1, 1), m, dtype="float"))
    return boom.Matrix(np.array(m, dtype="float"))


def to_boom_spd(m):
    """
    Convert a matrix-like object to boom.SpdMatrix (symmetric positive
    definite).  Accepts numpy arrays, pandas DataFrames, scalars, or an
    existing boom.SpdMatrix.
    """
    import BayesBoom.boom as boom
    if isinstance(m, boom.SpdMatrix):
        return m
    if hasattr(m, "values") and hasattr(m, "dtypes"):
        return boom.SpdMatrix(m.values.astype("float").copy())
    if isinstance(m, Number):
        return boom.SpdMatrix(np.full((1, 1), m, dtype="float"))
    return boom.SpdMatrix(np.array(m, dtype="float"))


def first_true(boolean_array):
    """Return the index of the first True element, or None if none found."""
    return next((i for i, v in enumerate(boolean_array) if v), None)


def unique_match(value, legal_value_list):
    """
    Return the element of legal_value_list that 'value' uniquely
    prefix-matches.  Returns None if there is no unique match.
    """
    matches = np.array([x.startswith(value) for x in legal_value_list])
    if matches.sum() != 1:
        return None
    return legal_value_list[first_true(matches)]
