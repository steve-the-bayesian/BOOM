import numpy as np
import pandas as pd
from numbers import Number


def _cbind_numpy(*args):
    """
    Collect all the arguments together in a single Matrix.  Scalar arguments
    are repeated to the implied number of rows.
    """
    args = [*args]
    shapes = [x.shape for x in args if not isinstance(x, Number)]
    if len(shapes) == 0:
        raise Exception("At least one array must be passed to cbind.")
    nrows = np.array([x[0] for x in shapes])
    if not np.all(nrows == nrows[0]):
        raise Exception("Incompatible shapes in cbind.")
    nrows = nrows[0]
    ncols = [x[1] for x in shapes if len(x) > 1]
    args = [np.full(nrows, x) if isinstance(x, Number) else x
            for x in args]
    if len(ncols) == 0:
        return np.array(args).T
    else:
        return np.hstack([x.reshape((nrows, -1)) for x in args])


def _cbind_pandas(*args):
    """
    Collect all the arguments together in a single data frame.  Scalar
    arguments are repeated to the implied number of rows.
    """
    args = [*args]
    shapes = [x.shape for x in args if not isinstance(x, Number)]
    if len(shapes) == 0:
        raise Exception("At least one array must be passed to cbind.")
    nrows = np.array([x[0] for x in shapes])
    if not np.all(nrows == nrows[0]):
        raise Exception("Incompatible shapes in cbind.")
    nrows = nrows[0]

    names = []
    pos = 0
    for i, arg in enumerate(args):
        if isinstance(arg, Number):
            names.append(f"X{pos + 1}")
            pos += 1
        elif isinstance(arg, np.ndarray):
            if len(arg.shape) == 1:
                names.append(f"X{pos + 1}")
                pos += 1
            elif len(arg.shape) == 2:
                names += [f"X{pos + i + 1}" for i in range(arg.shape[1])]
                pos += arg.shape[1]
            else:
                raise Exception(
                    "Can't call cbind on arrays with dimension more than 2")
        elif isinstance(arg, pd.Series):
            names.append(f"X{pos + 1}")
            pos += 1
        elif isinstance(arg, pd.DataFrame):
            names += list(arg.columns)

    args = [np.full(nrows, x) if isinstance(x, Number) else x
            for x in args]
    args = [pd.DataFrame(x) for x in args]

    ans = pd.concat(args, axis=1, ignore_index=True)
    ans.columns = names
    return ans


def cbind(*args):
    args = [*args]
    has_pandas = np.any([
        isinstance(x, pd.Series)
        or isinstance(x, pd.DataFrame)
        for x in args])
    if has_pandas:
        return _cbind_pandas(*args)
    else:
        return _cbind_numpy(*args)
