import bsts
from R import *
import pandas as pd
import numpy as np


data = pd.read_csv("~/airpass", index_col=0, sep=" ", names=["AirPassengers"])
data = np.log(data)


model = bsts.Bsts()
model.add_local_level(data)
model.add_seasonal(data, nseasons=12)

model.train(niter=100)
