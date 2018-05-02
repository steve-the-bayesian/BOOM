# Copyright 2018 Google LLC. All Rights Reserved.
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
library(testthat)

test_that("Shrinkage regression runs", {
  seed <- 8675309
  set.seed(seed)
  b0 <- -1
  b1 <- rnorm(20, 3, .2)
  b2 <- rnorm(30, -4, 7)
  nobs <- 10000
  beta <- c(b0, b1, b2)

  X <- cbind(1, matrix(rnorm(nobs * (length(beta) - 1)),
                       nrow = nobs,
                       ncol = length(beta) - 1))
  y.hat <- X %*% beta
  y <- rnorm(nobs, y.hat, .5)

  groups <- list(
    intercept = CoefficientGroup(1, prior = NormalPrior(0, 100)),
    first = CoefficientGroup(2:21,
                             mean.hyperprior = NormalPrior(0, 100),
                             sd.hyperprior = SdPrior(.2, 1)),
    second = CoefficientGroup(22:51,
                              mean.hyperprior = NormalPrior(0, 100),
                              sd.hyperprior = SdPrior(7, 1)))

  model <- ShrinkageRegression(y, X, groups,
                               residual.precision.prior = SdPrior(.5, 1),
                               niter = 1000,
                               ping = -1,
                               seed = seed)

  ## The intercept is in a singleton group. so its group mean won't update.
  expect_true(CheckMcmcMatrix(model$group.means[, -1], truth = c(3, -4)),
    info = paste("group.means...",
      McmcMatrixReport(model$group.means[, -1], truth = c(3, -4)),
      collapse = "\n"))
    
  ## The sd for the intercept is always zero, so don't check it here.
  expect_true(CheckMcmcMatrix(model$group.sds[, -1], truth = c(.2, 7)),
    info = paste("group sd's...",
      McmcMatrixReport(model$group.sds[, -1], truth = c(.2, 7)),
      collapse = "\n"))

  expect_true(CheckMcmcMatrix(model$coefficients, truth = beta),
    info = paste("coefficients...",
      McmcMatrixReport(model$coefficients, truth = beta),
      collapse = "\n"))

  expect_true(CheckMcmcVector(model$residual.sd, truth =.5),
    info = "residual.sd")
})
