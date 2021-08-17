# import unittest
import BayesBoom.dynreg as dr
import BayesBoom.spikeslab as ss

import numpy as np
import pandas as pd

import pdb

sample_size = 100
predictors = pd.DataFrame(np.random.randn(sample_size, 4))
response = np.sum(predictors, axis=1) + np.random.randn(sample_size)

data = pd.DataFrame()
data["y"] = response
for i in range(predictors.shape[1]):
    data[f"x{i+1}"] = predictors.iloc[:, i]
timestamps = np.array(["2007-10-13"] * sample_size).astype("datetime64[ns]")
data["timestamps"] = timestamps

model = dr.SparseDynamicRegressionModel(
    "y ~ " + ss.dot(data, ['y', 'timestamps']),
    data=data,
    timestamps="timestamps",
    niter=100)

print("all done!")
