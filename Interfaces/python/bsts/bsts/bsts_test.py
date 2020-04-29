import bsts
import numpy as np
import pdb

y = np.log(bsts.AirPassengers)

model = bsts.Bsts()
# model.add_state(bsts.LocalLevelStateModel(y))
model.add_state(bsts.LocalLinearTrendStateModel(y))

model.add_state(bsts.SeasonalStateModel(y, nseasons=12))
model.train(data=y, niter=100)
