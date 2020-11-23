import numpy as np
import pandas as pd


def simulate_data(sample_size: int, numeric_dim=1, cat_levels={}):
    numerics = np.random.randn(sample_size, numeric_dim)
    numeric_names = ["X" + str(i+1) for i in range(numeric_dim)]
    data = pd.DataFrame(numerics, columns=numeric_names)
    for vname, levels in cat_levels.items():
        data[vname] = np.random.choice(levels, size=sample_size)
    return data
