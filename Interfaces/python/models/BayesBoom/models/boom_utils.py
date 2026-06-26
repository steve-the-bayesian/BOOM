"""
Low-level utilities for converting Python/NumPy objects to their BayesBoom
C++ counterparts.  These functions are intentionally kept self-contained so
that BayesBoom.models does not require BayesBoom.R at import time.
"""

import numpy as np
import pandas as pd
from numbers import Number
from pandas.api.types import (
    is_numeric_dtype,
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_object_dtype,
)


def is_all_numeric(data_frame):
    """Return True iff every column of data_frame has a numeric dtype."""
    return np.all(data_frame.dtypes.apply(is_numeric_dtype))


def is_iterable(obj):
    """Return True if obj supports iteration (has __getitem__ or __iter__)."""
    return hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")


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
    if hasattr(m, "values") and hasattr(m, "dtypes") and is_all_numeric(m):
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
    if hasattr(m, "values") and hasattr(m, "dtypes") and is_all_numeric(m):
        return boom.SpdMatrix(m.values.astype("float").copy())
    if isinstance(m, Number):
        return boom.SpdMatrix(np.full((1, 1), m, dtype="float"))
    return boom.SpdMatrix(np.array(m, dtype="float"))


def to_boom_labelled_matrix(data_frame):
    """Convert a pandas DataFrame to a boom.LabelledMatrix."""
    import BayesBoom.boom as boom
    return boom.LablledMatrix(data_frame.values.astype("float"),
                              data_frame.index.astype("str"),
                              data_frame.columns.astype("str"))


def to_boom_array(arr):
    """
    Convert a multi-way numpy array to a boom.Array object.  Converts from
    C-style (row-major) to BOOM's column-major storage order.
    """
    import BayesBoom.boom as boom
    if not isinstance(arr, np.ndarray) or arr.dtype != "float":
        arr = np.array(arr, dtype="float", order="F")
    return boom.Array(arr.shape, to_boom_vector(arr.reshape(-1, order="F")))


def to_numpy(x):
    """Convert a boom or pandas object to a numpy array."""
    import BayesBoom.boom as boom
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (pd.Series, pd.DataFrame)):
        return x.values
    elif isinstance(x, (boom.Vector, boom.Matrix, boom.SpdMatrix, boom.Array)):
        return x.to_numpy()
    else:
        return np.array(x)


def to_boom_date(timestamp):
    """Convert a single pd.Timestamp or similar to a boom.Date object."""
    import BayesBoom.boom as boom
    if isinstance(timestamp, str):
        timestamp = pd.Timestamp(timestamp)
    return boom._date(int(timestamp.month), int(timestamp.day),
                      int(timestamp.year))


def to_pd_timestamp(boom_date):
    """Convert a boom.Date or boom.DateTime to a pd.Timestamp."""
    import BayesBoom.boom as boom
    if isinstance(boom_date, boom.Date):
        return pd.Timestamp(month=boom_date.month,
                            day=boom_date.day,
                            year=boom_date.year)
    elif isinstance(boom_date, boom.DateTime):
        return pd.Timestamp(month=boom_date.month,
                            day=boom_date.day,
                            year=boom_date.year,
                            hour=boom_date.hour,
                            minute=boom_date.minute,
                            second=boom_date.second,
                            nanoseconds=boom_date.nanosecond)
    else:
        raise Exception("Wrong input type.")


def to_boom_datetime_vector(series):
    """Convert a pandas datetime Series to a boom datetime vector."""
    import BayesBoom.boom as boom
    series = pd.Series(pd.to_datetime(series))

    year = series.dt.year.tolist()
    month = series.dt.month.tolist()
    day = series.dt.day.tolist()

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

    return boom.to_boom_datetime_vector(year, month, day,
                                        to_boom_vector(day_fraction))


def to_pd_datetime64(boom_datetime_vector):
    """
    Convert a vector of boom.DateTime objects to a pd.Series of dtype
    datetime64[ns].
    """
    import BayesBoom.boom as boom
    ns = boom.to_nanoseconds(boom_datetime_vector)
    return pd.Series(np.array(ns, dtype="datetime64[ns]"))


def to_boom_data_table(data: pd.DataFrame):
    """
    Create a boom.DataTable from a pandas DataFrame.

    dtype mapping:
      numeric / bool   -> numeric
      datetime64       -> datetime
      object / string / categorical -> categorical or high_cardinality
    """
    import BayesBoom.boom as boom
    from BayesBoom.R.summary import CategoricalSummary
    dtypes = data.dtypes
    ans = boom.DataTable()
    for i in range(data.shape[1]):
        dt = dtypes.iloc[i]
        vname = data.columns[i]
        y = data.iloc[:, i]
        if is_numeric_dtype(dt) or is_bool_dtype(dt):
            ans.add_numeric(boom.Vector(y.values.astype("float")), vname)
        elif is_datetime64_any_dtype(dt):
            ans.add_datetime(to_boom_datetime_vector(y), vname)
        elif (
            isinstance(dt, pd.CategoricalDtype)
            or isinstance(dt, pd.StringDtype)
            or is_object_dtype(dt)
        ):
            labels = y.astype(str)
            cat_summary = CategoricalSummary(labels, max_levels=None)
            if cat_summary.is_high_cardinality:
                ans.add_high_cardinality(labels.tolist(), vname)
            else:
                ans.add_categorical_from_labels(labels.values, vname)
        else:
            raise Exception(
                f"Only numeric, categorical, string, or datetime data are "
                f"supported.  Column {i} ({data.columns[i]}) has dtype {dt}."
            )
    return ans


def to_boom_mixed_data(row) -> "boom.MixedMultivariateData":
    """
    Convert a single-row pandas DataFrame or Series to a
    boom.MixedMultivariateData.
    """
    if isinstance(row, pd.Series):
        row = row.to_frame().T
    if len(row) != 1:
        raise ValueError(
            f"Expected a single-row DataFrame, got {len(row)} rows.")
    return to_boom_data_table(row).row(0)


def _boom_labelled_matrix_to_pd_dataframe(boom_labelled_matrix):
    """Convert a boom.LabelledMatrix to a pandas DataFrame."""
    idx = boom_labelled_matrix.row_names
    cols = boom_labelled_matrix.col_names
    values = boom_labelled_matrix.to_numpy()

    if not idx:
        idx = np.arange(values.shape[0])
    if not cols:
        cols = np.arange(values.shape[1])

    return pd.DataFrame(values, idx, cols)


def _boom_data_table_to_pd_dataframe(data, columns=None, index=None):
    """Convert a boom.DataTable to a pd.DataFrame."""
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
        elif vtype == "high_cardinality":
            ans_as_dict[vname] = pd.array(
                data.get_high_cardinality(i), dtype=pd.StringDtype())
        else:
            raise Exception(
                f"Unrecognized variable type '{vtype}' for column {i} "
                f"({vname}).")

    return pd.DataFrame(ans_as_dict, index=index)


def to_pd_dataframe(obj, columns=None, index=None):
    """Convert a boom.LabelledMatrix or boom.DataTable to a pd.DataFrame."""
    import BayesBoom.boom as boom
    if isinstance(obj, boom.LabelledMatrix):
        return _boom_labelled_matrix_to_pd_dataframe(obj)
    elif isinstance(obj, boom.DataTable):
        return _boom_data_table_to_pd_dataframe(obj, columns, index)
    else:
        raise Exception(f"Unrecognized type {type(obj)} passed to "
                        "'to_pd_dataframe'.")


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
