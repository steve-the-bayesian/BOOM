import BayesBoom.mixtures as mix
import numpy as np

y1 = np.random.randn(100, 3)
y2 = np.random.randn(150, 3) + 4
y = np.concatenate((y1, y2), axis=0)

model = mix.DirichletProcessMvn(y)
model.mcmc(10000)
