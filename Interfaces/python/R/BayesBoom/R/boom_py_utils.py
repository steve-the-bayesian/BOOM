# This module contains low-level utilities for things like type-checking and
# coercion.

import numpy as np
import pandas as pd
import BayesBoom.boom as boom
from numbers import Number


def is_all_numeric(data_frame):
    """
    Returns True iff 'data_frame' is a pd.DataFrame (or equivlalent)
    """
    return np.all(data_frame.dtypes.apply(pd.api.types.is_numeric_dtype))


def is_iterable(object):
    """
    Check to see if 'object' is an iterable object.  An iterable object allows
    the idiom 'for x in object:...'.
    """
    return hasattr(object, "__getitem__") or hasattr(object, "__iter__")


def to_boom_vector(v):
    """
    Convert the vector-like object 'v' to a boom.Vector.  This is a more user
    friendly experience than relying on the boom.Vector constructor, which only
    accepts floating point numpy arrays.  Here 'v' can be a numeric scalar, a
    numpy array of any numeric dtype, a pandas Series of any numeric dtype, or
    any similar object that either acts like a pd.Series or is convertible to a
    np.array.
    """
    if hasattr(v, "values"):
        # Handle pd.Series and similar.
        return boom.Vector(np.array(v.values, dtype="float"))

    if isinstance(v, Number):
        return boom.Vector(np.array([v], dtype="float"))

    return boom.Vector(np.array(v, dtype="float"))


def to_boom_matrix(m):
    """
    Convert the matrix-like object 'm' to a boom.Matrix.  This is a more user
    friendly experience than relying on the boom.Matrix constructor, which only
    accepts floating point numpy arrays.  Here 'm' can be a numeric scalar, a
    numpy array of any numeric dtype, a pandas DataFrame containing numeric
    data, or any similar object that either acts like a pd.DataFrame or is
    convertible to a np.array.
    """
    if hasattr(m, "values") and hasattr(m, "dtypes") and is_all_numeric(m):
        # Handle pd.DataFrame and similar.
        return boom.Matrix(m.values.astype("float"))

    if isinstance(m, Number):
        return boom.Matrix(np.full((1, 1), m, dtype="float"))

    return boom.Matrix(np.array(m, dtype="float"))


def to_boom_spd(m):
    """
    Convert the matrix-like object 'm' to a boom.Matrix.  This is a more user
    friendly experience than relying on the boom.Matrix constructor, which only
    accepts floating point numpy arrays.  Here 'm' can be a numeric scalar, a
    numpy array of any numeric dtype, a pandas DataFrame containing numeric
    data, or any similar object that either acts like a pd.DataFrame or is
    convertible to a np.array.
    """
    if hasattr(m, "values") and hasattr(m, "dtypes") and is_all_numeric(m):
        # Handle pd.DataFrame and similar.
        return boom.SpdMatrix(m.values.astype("float"))

    if isinstance(m, Number):
        return boom.SpdMatrix(np.full((1, 1), m, dtype="float"))

    return boom.SpdMatrix(np.array(m, dtype="float"))


def to_numpy(x):
    """
    Convert x to a numpy array.
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    elif isinstance(x, (boom.Vector, boom.Matrix, boom.SpdMatrix)):
        return x.to_numpy()
    else:
        return np.array(x)


def to_boom_date(timestamp):
    """
    Convert a pd.Timestamp or similar to a boom.Date object.
    """
    if isinstance(timestamp, str):
        timestamp = pd.Timestamp(timestamp)

    return boom._date(int(timestamp.month), int(timestamp.day),
                      int(timestamp.year))


def to_pd_timestamp(boom_date):
    """
    Convert a boom.Date or boom.DateTime to a pd.Timestamp.
    """
    if isinstance(boom_date, boom.Date):
        return pd.Timestamp(month=boom_date.month,
                            day=boom_date.day,
                            year=boom_date.year)
    elif isinstance(boom_date, boom.DateTime):
        pass
    else:
        raise Exception("Wrong input type.")
