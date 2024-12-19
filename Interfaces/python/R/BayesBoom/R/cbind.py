import numpy as np
import pandas as pd
from numbers import Number
import inspect
import pdb

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
    for i, arg in enumerate(args):
        # Get the name used to pass 'arg' into the function.  This code gets
        # the current frame, moves back up two frames (to the frame that
        # called cbind) and then checks the pointer to 'arg' (using 'is' and
        # not '==' for comparison) against everything in that frame.
        #
        # This can fail if there is more than one name pointing to the same
        # memory in the calling frame.  
        callers_local_vars = inspect.currentframe().f_back.f_back.f_locals.items()
        try:
            argname = next(var_name for var_name, var_val in callers_local_vars
                           if var_val is arg)
        except StopIteration:
            argname = None
            
        if isinstance(arg, Number):
            # If a number was passed as an argument, use the value of that
            # number as a column heading if no name is found in the calling
            # environment.
            names.append(argname if argname else str(arg))
        elif isinstance(arg, np.ndarray):
            if len(arg.shape) == 1:
                # If a 1-dimensional numpy array is found, use the name of that
                # numpy array as column heading.
                names.append(argname)
            elif len(arg.shape) == 2:
                # If a 2D numpy array is found, then use the name of the array
                # as column headings, with column numbers appended for each
                # column.
                names += [argname + str(i) for i in range(arg.shape[1])]
            else:
                raise Exception(
                    "Can't call cbind on arrays with dimension more than 2")
                
        elif isinstance(arg, pd.Series):
            names.append(argname)

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
