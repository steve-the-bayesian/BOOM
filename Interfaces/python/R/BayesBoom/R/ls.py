import numpy as np
import pandas as pd
from .pretty import pretty
from inspect import isfunction, getsource


def ls(*args, hide_underscore=True, maxwidth=80):
    """
    List the contents of one or more objects.  If passed a function then print
    the body of the function.
    """

    if len([*args]) == 0:
        print("If you're trying to list the global namespace type "
              "'pretty(dir())'.")
        print("(The interactive namespace is not available to modules.)")
#        pretty(sorted(globals()), hide_underscore=hide_underscore)
    elif len([*args]) == 1 and isfunction(args[0]):
        print(getsource(args[0]))
    else:
        for arg in args:
            pretty(dir(arg), hide_underscore=hide_underscore, width=maxwidth)
            print("\n")


def ls_object(obj):
    """
    List out the contents of an object, noting which elements are
      - functions       F,
      - dicts           D,
      - lists,          L,
      - numpy arrays    NP,
      - pandas objects  PD,
      - other objects   O,

    For containers, list the size of the container.  For numpy or pandas objects
    list the object's shape.
    """

    # contents = dir(obj)
    # functions = [x for x in contents if isfunction(getattr(obj, x))]
    # dicts = [x for x in contents if isinstance(getattr(obj, x), dict)]
    # lists = [x for x in contents if isinstance(getattr(obj, x), list)]
    # numpy_arrays = [x for x in contents
    #                 if isinstance(getattr(obj, x), np.ndarray)]
    # pandas = [x for x in contents if isinstance(getattr(obj, x), pd.Series)
    #           or isinstance(getattr(obj, x), pd.DataFrame)]
    # other = [x for x in contents
    #          if x not in functions
    #          and x not in dicts
    #          and x not in lists
    #          and x not in numpy_arrays
    #          and x not in pandas]

    lists = [3]
    list_sizes = [x + f"[{len(getattr(obj, x))}]" for x in lists]
    return list_sizes
