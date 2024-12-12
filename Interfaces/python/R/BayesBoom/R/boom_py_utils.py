# This module contains low-level utilities for things like type-checking and
# coercion.

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype,
    is_object_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype
)

import BayesBoom.boom as boom
from numbers import Number



def is_all_numeric(data_frame):
    """
    Returns True iff 'data_frame' is a pd.DataFrame (or equivlalent)
    """
    return np.all(data_frame.dtypes.apply(is_numeric_dtype))


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
    Convert a single pd.Timestamp or similar to a boom.Date object.
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
        return pd.Timestamp(month = boom_date.month,
                            day = boom_date.day,
                            year = boom_date.year,
                            hour = boom_date.hour,
                            minute = boom_date.minute,
                            second = boom_date.second,
                            nanoseconds= boom_date.nanosecond)

    else:
        raise Exception("Wrong input type.")


def _boom_labelled_matrix_to_pd_dataframe(boom_labelled_matrix):
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
    series = pd.Series(pd.to_datetime(series))

    # convert dates to lists of years, months, and days (integers).
    year = series.dt.year.tolist()
    month = series.dt.month.tolist()
    day = series.dt.day.tolist()

    # The number of nanoseconds in each time unit.
    NANO = 1.0
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


def to_pd_datetime64(boom_datetime_vector):
    """
    Convert a vector (list) of boom.DateTime objects into a pd.Series of dtype
    "datetime64[ns]".
    """
    ns = boom.to_nanoseconds(boom_datetime_vector)
    return pd.Series(np.array(ns, dtype="datetime64[ns]"))


def to_boom_data_table(data: pd.DataFrame):
    """
    Create a BOOM DataTable object from a pandas DataFrame.  The categories of
    any categorical variables will be handled as strings.
    """
    dtypes = data.dtypes
    ans = boom.DataTable()
    for i in range(data.shape[1]):
        dt = dtypes.iloc[i]
        vname = data.columns[i]
        y = data.iloc[:, i]
        if is_numeric_dtype(dt) or is_bool_dtype(dt):
            ans.add_numeric(boom.Vector(y.values.astype("float")),
                            vname)
        elif isinstance(dt, pd.CategoricalDtype):
            # Note the pandas function is_categorical_dtype is deprecated in
            # favor of the isinstance call above.
            ans.add_categorical(y.cat.codes, y.cat.categories, vname)
        elif is_object_dtype(dt):
            labels = y.astype("str")
            ans.add_categorical_from_labels(labels.values, vname)
        elif is_datetime64_any_dtype(dt):
            ans.add_datetime(
                to_boom_datetime_vector(y),
                vname)
        else:
            raise Exception(
                f"Only numeric, categorical, or datetime data are supported.  "
                f"Column {i} ({data.columns[i]}) has dtype {dt}."
            )
    return ans


def _boom_data_table_to_pd_dataframe(data: boom.DataTable, columns=None, index=None):
    """
    Convert a boom.DataTable to a pd.DataFrame.

    Args:
      data:  The data table to be converted.
      columns: The column names.  If None then the names of the data table will
        be used.
      index: The index to be applied to the returned data frame.  If None then
        a numeric range will be used.

    Returns:
      A pandas data frame containing the data from 'data'.
    """
    if columns is None:
        columns = data.variable_names

    if len(columns) == 0:
        columns = ["V" + str(i) for i in range(data.ncol)]

    if len(columns) != data.ncol:
        raise Exception("The number of entries in 'columns' must match "
                        "the number of variables in the data table.")

    if index is None:
        index = range(data.nrow)
    if len(index) != data.nrow:
        raise Exception("The number of entries in 'index' must match "
                        "the number of rows in the data table.")

    ans_as_dict = {}
    for i, vname in enumerate(columns):
        vtype = data.variable_type(i)
        if vtype == "numeric":
            ans_as_dict[vname] = data.getvar(i).to_numpy()
        elif vtype == "categorical":
            values = data.get_nominal_values(i)
            levels = data.get_nominal_levels(i)
            ans_as_dict[vname] = pd.Categorical.from_codes(values, levels)
        elif vtype == "datetime":
            ans_as_dict[vname] = to_pd_datetime64(data.get_datetime(i))
        else:
            raise Exception("Only numeric or categorical values are supported.")

    return pd.DataFrame(ans_as_dict, index=index)


def to_pd_dataframe(obj, columns=None, index=None):
    """
    Convert a boom object to to a pd.DataFrame.
    """
    if isinstance(obj, boom.LabelledMatrix):
        return _boom_labelled_matrix_to_pd_dataframe(obj)
    elif isinstance(obj, boom.DataTable):
        return _boom_data_table_to_pd_dataframe(obj, columns, index)
    else:
        raise Exception(f"Unrecognized type {type(obj)} passed "
                        "to 'to_pd_dataframe'.")
