import numpy as np
import pandas as pd


def simulate_data(sample_size: int,
                  numeric_dim=1,
                  cat_levels={},
                  date_fields={}):
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
        and end.

    Returns:
      A pd.DataFrame with the requested columns.

    Example Usage:
      data = simulate_data(100, numeric_dim=3, cat_levels={
        "color": ["red", "blue", "green"],
        "stooges": ["Larry", "Moe", "Curly", "Shemp"]
      })

      # sample output:
        >>> data
                  X1        X2        X3  color stooges
        0   1.894207  0.735859 -0.870642  green   Shemp
        1  -0.726997 -1.189791  2.113498   blue   Shemp
        2   0.389256  1.810293 -1.791016  green   Curly
        3   1.615793 -0.319811 -0.943746   blue   Curly
        4   0.252074  0.244390  0.987215   blue   Shemp
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
        
    return data
