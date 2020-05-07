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

cat("test-nnet\n")

data(BostonHousing)
hidden.layers <- list(
  HiddenLayer(10, expected.model.size = Inf))
nnet.niter <- 1000
model <- BayesNnet(medv ~ ., hidden.layers = hidden.layers, niter = nnet.niter, data = BostonHousing, expected.model.size = Inf, seed = seed)

reg.niter <- 1000
reg <- lm.spike(medv ~ ., niter = reg.niter, data = BostonHousing, seed=seed)
reg.burn <- SuggestBurnLogLikelihood(-reg$sigma)
nnet.burn <- SuggestBurnLogLikelihood(-model$residual.sd)

if (reg.burn < .9 * reg.niter && nnet.burn < .9 * nnet.niter) {
  expect_gt(mean(reg$sigma[reg.burn:reg.niter]),
    mean(model$residual.sd[nnet.burn:nnet.niter]))
}

pred <- predict(model)
plot(model)
plot(model, "resid")
plot(model, "structure")
