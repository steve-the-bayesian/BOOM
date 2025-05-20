import numpy as np
import pandas as pd
from .random_strings import random_strings

def simulate_data(sample_size: int,
                  numeric_dim=1,
                  cat_levels={},
                  date_fields={},
                  high_cardinality_fields={}):
    """
    Simulate a pandas DataFrame object with a mix of numeric and categorical
    variables.  The intent is that the simulated data be used in unit tests.

    Args:
      sample_size:  The desired number of rows in the simulated data frame.
    
      numeric_dim: The desired number of numeric columns.  These are labeled
        X1, X2, ... .

      cat_levels: Column names (keys) and levels (values) for the categorical
        variables to be simulated.
    
      date_fields: Column names (keys) and a pair (tuple) of (begin, end) dates.
        The data will be populated by a random selection of dates between begin
        and end.   The dates are not sorted.

      high_cardinality_fields: Column names (keys) and an integer specifying
        string lengths.  The data will be populated by random strings of the
        given length.  Each entry is likely to be unique, but a small set of
        repeated values is possible.

    Returns:
      A pd.DataFrame with the requested columns.

    Example Usage:
    data = simulate_data(
        100,
        numeric_dim=3,
        cat_levels={
            "color": ["red", "blue", "green"],
            "stooges": ["Larry", "Moe", "Curly", "Shemp"]
        },
        date_fields={
            "birthday": ("1970-01-01", "1980-3-10")
        },
        high_cardinality_fields={
            "user_id": 7
        }
    )

      # sample output:
        >>> data
              X1        X2        X3      color stooges birthday    user_id
        0  -0.352878 -0.119473 -0.668666  green     Moe 1977-01-16  ijvudnf
        1  -0.100375  0.664306  0.647133  green     Moe 1974-11-17  zsqyfds
        2  -0.076087 -0.045390 -0.377485   blue     Moe 1972-09-18  wqhbkna
        3  -0.189324  0.771207 -0.757603  green   Larry 1974-05-11  nyctvdh
        4  -0.035751  0.894994  0.306342  green   Larry 1978-12-27  rsjhxcj
        ...
    """
    
    numerics = np.random.randn(sample_size, numeric_dim)
    numeric_names = ["X" + str(i+1) for i in range(numeric_dim)]
    data = pd.DataFrame(numerics, columns=numeric_names)
    
    for vname, levels in cat_levels.items():
        data[vname] = np.random.choice(levels, size=sample_size)

    for vname, begin_end in date_fields.items():
        begin = pd.to_datetime(begin_end[0])
        end = pd.to_datetime(begin_end[1])

        num_days = (end - begin).days + 1  # include the end day

        dates = begin + pd.to_timedelta(
            np.random.randint(0, num_days, size=sample_size),
            unit="days"
        )
        data[vname] = dates

    for vname, string_size in high_cardinality_fields.items():
        data[vname] = random_strings(sample_size,
                                     string_size,
                                     ensure_unique=False)
        
    return data
