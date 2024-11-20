# This module contains low-level utilities for things like type-checking and
# coercion.

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype
)

import BayesBoom.boom as boom
from numbers import Number



def is_all_numeric(data_frame):
    """
    Returns True iff 'data_frame' is a pd.DataFrame (or equivlalent)
    """
    return np.all(data_frame.dtypes.apply(pd.api.types.is_numeric_dtype))


def is_iterable(obj):
    """
    Check to see if 'object' is an iterable object.  An iterable object allows
    the idiom 'for x in object:...'.
    """
    return hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")


def to_boom_vector(v):
    """
    Convert the vector-like object 'v' to a boom.Vector.  This is a more user
    friendly experience than relying on the boom.Vector constructor, which only
    accepts floating point numpy arrays.  Here 'v' can be a numeric scalar, a
    numpy array of any numeric dtype, a pandas Series of any numeric dtype, or
    any similar object that either acts like a pd.Series or is convertible to a
    np.array.
    """
    if isinstance(v, boom.Vector):
        return v

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
    if isinstance(m, boom.Matrix):
        return m

    if hasattr(m, "values") and hasattr(m, "dtypes") and is_all_numeric(m):
        # Handle pd.DataFrame and similar.
        return boom.Matrix(m.values.astype("float"))

    if isinstance(m, Number):
        return boom.Matrix(np.full((1, 1), m, dtype="float"))

    return boom.Matrix(np.array(m, dtype="float"))


def to_boom_labelled_matrix(data_frame):
    """
    Convert a pandas data frame to a boom.LablledMatrix.
    """
    return boom.LablledMatrix(data_frame.values.astype("float"),
                              data_frame.index.astype("str"),
                              data_frame.columns.astype("str"))


def to_boom_spd(m):
    """
    Convert the matrix-like object 'm' to a boom.Matrix.  This is a more user
    friendly experience than relying on the boom.Matrix constructor, which only
    accepts floating point numpy arrays.  Here 'm' can be a numeric scalar, a
    numpy array of any numeric dtype, a pandas DataFrame containing numeric
    data, or any similar object that either acts like a pd.DataFrame or is
    convertible to a np.array.
    """
    if isinstance(m, boom.SpdMatrix):
        return m

    if hasattr(m, "values") and hasattr(m, "dtypes") and is_all_numeric(m):
        # Handle pd.DataFrame and similar.
        return boom.SpdMatrix(m.values.astype("float"))

    if isinstance(m, Number):
        return boom.SpdMatrix(np.full((1, 1), m, dtype="float"))

    return boom.SpdMatrix(np.array(m, dtype="float"))

def to_boom_array(arr):
    """
    Convert a multi-way numpy array to a boom.Array object.  By default numpy
    stores array in C-style "row major" order, while BOOM expects arrays in
    "column major" order.
    """
    if (
            not isinstance(arr, np.ndarray)
            or arr.dtype != "float"
    ):
        arr = np.array(arr, dtype="float", order="F")

    return boom.Array(arr.shape,
                      to_boom_vector(arr.reshape(-1, order="F")))

def to_numpy(x):
    """
    Convert x to a numpy array.
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    elif isinstance(x, (boom.Vector, boom.Matrix, boom.SpdMatrix, boom.Array)):
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


def to_pd_dataframe(boom_labelled_matrix):
    """
    Convert a boom.LabelledMatrix to a pandas DataFrame.
    """

    idx = boom_labelled_matrix.row_names
    cols = boom_labelled_matrix.col_names
    values = boom_labelled_matrix.to_numpy()

    if not idx:
        idx = np.arange(values.shape[0])

    if not cols:
        cols = np.arange(values.shape[1])

    return pd.DataFrame(values, idx, cols)


def to_boom_datetime_vector(series):
    series = pd.to_datetime(series)

    # convert dates to lists of years, months, and days (integers).
    year = series.dt.year.tolist()
    month = series.dt.month.tolist()
    day = series.dt.day.tolist()

    # If we measure the world in nanoseconds, then
    MICRO = 1000.0
    SECOND = MICRO * 1000.0 * 1000.0
    MINUTE = 60.0 * SECOND
    HOUR = 60.0 * MINUTE
    DAY = HOUR * 24.0

    day_fraction = (
        series.dt.hour * HOUR
        + series.dt.minute * MINUTE
        + series.dt.second * SECOND
        + series.dt.microsecond * MICRO
        + series.dt.nanosecond
    ) / DAY

    return boom.to_boom_datetime_vector(
        year,
        month,
        day,
        to_boom_vector(day_fraction))


def to_boom_data_table(pandas_df):
    """
    Convert a pandas data frame to a boom.DataTable.
    """
    ans = boom.DataTable()
    for column_name in pandas_df.columns:
        colname = str(column_name)
        y = pandas_df.loc[:, column_name]
        if is_numeric_dtype(y):
            ans.add_numeric(to_boom_vector(y), colname)
        elif is_object_dtype(y):
            ans.add_categorical_from_labels(y, colname)
        elif isinstance(y.dtype, pd.CategoricalDtype):
            # Note the pandas function is_categorical_dtype is deprecated in
            # favor of the isinstance call above.
            levels = y.cat.categories.tolist()
            values = y.cat.codes.tolist()
            ans.add_categorical(values, levels, colname)
        elif is_datetime64_any_dtype(y.dtype):
            ans.add_datetime(
                R.to_boom_datetime_vector(y),
                colname)
        else:
            raise Exception(f"unspported dtype in column {colname}.")
