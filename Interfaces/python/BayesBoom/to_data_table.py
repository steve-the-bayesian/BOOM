import pandas as pd
from pandas.api.types import is_numeric_dtype


def to_data_table(data: pd.DataFrame):
    import BayesBoom.boom as boom
    table = boom.DataTable()
    for vname in data.columns:
        if is_numeric_dtype(data[vname]):
            table.add_numeric(boom.Vector(data[vname].values.astype("float")),
                              str(vname))
        else:
            variable = data[vname].astype("category")
            categories = variable.cat.categories.tolist()
            table.add_categorical(
                variable.cat.codes.values,
                [str(x) for x in categories],
                vname,
            )
    return table


# def to_pandas(data):
#     import BayesBoom.boom as boom
#     nrow = data.nrow
#     ncol = data.ncol
#     values = {}
#     vnames = data.variable_names
#     for i in range(ncol):
#         dtype_str = data.variable_type(i)
#         if dtype_str == "numeric":
#             values[vnames[i]] = pd.Series(data.getvar(i), dtype="float")
#         elif dtype_str == "categorical":
#             codes = data.get_nominal_values(i)
#             categories = data.get_nominal_levels(i)
