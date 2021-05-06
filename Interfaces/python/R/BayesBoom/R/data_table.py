import BayesBoom.boom as boom
import pandas as pd
from pandas.api.types import (
    is_numeric_dtype, is_categorical_dtype, is_object_dtype, is_bool_dtype
)


def to_data_table(data: pd.DataFrame):
    """
    Create a BOOM DataTable object from a pandas DataFrame.  The categories of
    any categorical variables will be handled as strings.
    """
    dtypes = data.dtypes
    ans = boom.DataTable()
    for i in range(data.shape[1]):
        dt = dtypes[i]
        vname = data.columns[i]
        if is_numeric_dtype(dt) or is_bool_dtype(dt):
            ans.add_numeric(boom.Vector(data.iloc[:, i].values.astype("float")),
                            vname)
        elif is_categorical_dtype(dt):
            x = data.iloc[:, i]
            values = x.cat.codes
            codes = x.cat.categories
            ans.add_categorical(values, codes, vname)
        elif is_object_dtype(dt):
            labels = data.iloc[:, i].astype("str")
            ans.add_categorical_from_labels(labels.values, vname)
        else:
            raise Exception(
                f"Only numeric or categorical data are supported.  "
                f"Column {i} ({data.columns[i]}) has dtype {dt}."
            )
    return ans


def to_data_frame(data: boom.DataTable, columns=None, index=None):
    """
    Convert a boom.DataTable to a pd.DataFrame.

    Args:
      data:  The data table to be converted.
      columns: The column names.  If None then the names of the data table will
        be used.
      index:  The index to be applied to the returned data frame.

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
        else:
            raise Exception("Only numeric or categorical values are supported.")

    return pd.DataFrame(ans_as_dict, index=index)
