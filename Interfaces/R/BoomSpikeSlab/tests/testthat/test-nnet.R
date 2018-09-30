# Copyright 2018 Steven L. Scott. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

library(BoomSpikeSlab)
# if (require(testthat) && require(mlbench)) {
require(testthat)
require(mlbench)
seed <- 8675309
set.seed(seed)
data(BostonHousing)
hidden.layers <- list(
  HiddenLayer(10, expected.model.size = Inf))
model <- BayesNnet(medv ~ ., hidden.layers = hidden.layers,
  niter = 1000, data = BostonHousing, expected.model.size = Inf, seed = seed)

reg <- lm.spike(medv ~ ., niter = 1000, data = BostonHousing)
reg.burn <- SuggestBurnLogLikelihood(-reg$sigma)
nnet.burn <- SuggestBurnLogLikelihood(-model$residual.sd)
expect_gt(mean(reg$sigma[reg.burn:1000]),
  mean(model$residual.sd[nnet.burn:1000]))

pred <- predict(model)
plot(model)
plot(model, "resid")
plot(model, "structure")

hidden.layers <- list(
  HiddenLayer(8, expected.model.size = Inf),
  HiddenLayer(8, expected.model.size = Inf))
deep.model <- BayesNnet(medv ~ ., hidden.layers = hidden.layers, niter = 1000,
  data = BostonHousing, expected.model.size = Inf, seed = seed)
deep.burn <- SuggestBurnLogLikelihood(-deep.model$residual.sd)
expect_gt(mean(model$residual.sd[nnet.burn:1000]),
  mean(deep.model$residual.sd[deep.burn:1000]))

