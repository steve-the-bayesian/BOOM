import BayesBoom as boom
# from BayesBoom.spikeslab import lm_spike
import numpy as np
import pandas as pd
import pdb

sample_size = 10000
ngood = 5
nbad = 30
x = np.random.randn(sample_size, ngood + nbad)

beta = np.array([1.2, .8, 2.7])
beta = np.random.randn(ngood) * 4

b0 = 7.2
residual_sd = .3
yhat = b0 + x[:, :ngood] @ beta
errors = np.random.randn(sample_size) * residual_sd
y = yhat + errors

data = pd.DataFrame(x, columns = ["X" + str(i) for i in range(x.shape[1])])
x_formula = "+".join(x for x in data.columns)
data["y"] = y
formula = f"y ~ {x_formula}"

model = lm_spike(formula, niter=100, data=data)
