import numpy as np
import bsts
import matplotlib.pyplot as plt
import pdb

y = np.log(bsts.AirPassengers)

model = bsts.Bsts()

# model.add_state(bsts.LocalLevelStateModel(y))
model.add_state(bsts.SemilocalLinearTrendStateModel(y))
model.add_state(bsts.SeasonalStateModel(y, nseasons=12))
model.train(data=y, niter=100)
# model.plot()
# plt.show()

model.plot("comp")

plt.show()
