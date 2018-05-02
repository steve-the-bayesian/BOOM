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

seed <- 8675309
set.seed(seed)

SimulateNestedRegressionData <- function() {
  beta.hyperprior.mean <- c(8, 6, 7, 5)
  xdim <- length(beta.hyperprior.mean)
  beta.hyperprior.variance <- rWishart(2 * xdim, diag(rep(1, xdim)),
                                       inverse = TRUE)

  number.of.groups <- 27
  nobs.per.group = 23
  beta <- rmvn(number.of.groups,
               beta.hyperprior.mean,
               beta.hyperprior.variance)

  residual.sd <- 2.4
  X <- cbind(1, matrix(rnorm(number.of.groups * (xdim - 1) * nobs.per.group),
                       ncol = xdim - 1))
  group.id <- rep(1:number.of.groups, len = nrow(X))
  y.hat <- numeric(nrow(X))
  for (i in 1:nrow(X)) {
    y.hat[i] = sum(X[i, ] * beta[group.id[i], ])
  }
  y <- rnorm(length(y.hat), y.hat, residual.sd)
  suf <- BoomSpikeSlab:::.RegressionSufList(X, y, group.id)

  return(list(beta.hyperprior.mean = beta.hyperprior.mean,
              beta.hyperprior.variance = beta.hyperprior.variance,
              beta = beta,
              residual.sd = residual.sd,
              X = X,
              y = y,
              group.id = group.id,
              suf = suf))
}

test_that("NestedRegression works", {
  ## Check that things work with default priors.
  d <- SimulateNestedRegressionData()
  xdim <- length(d$beta.hyperprior.mean)
  model <- NestedRegression(
      suf = d$suf,
      niter = 1000,
      seed = seed)

  expect_true(CheckMcmcMatrix(model$prior.mean, truth = d$beta.hyperprior.mean),
    info = McmcMatrixReport(model$prior.mean, truth = d$beta.hyperprior.mean))
  CheckMcmcVector(model$residual.sd, truth = d$residual.sd)
  xdim <- length(d$beta.hyperprior.mean)
  number.of.groups <- nrow(d$beta)
  for (v in 1:xdim) {
    expect_true(
      CheckMcmcMatrix(model$prior.variance[, v, ],
        truth = d$beta.hyperprior.variance[v, ]),
      info = McmcMatrixReport(model$prior.variance[, v, ],
        truth = d$beta.hyperprior.variance[v, ]))
  }
  for (g in 1:number.of.groups) {
    expect_true(
      CheckMcmcMatrix(model$coefficients[, g, ], truth = d$beta[g, ]),
      info = McmcMatrixReport(model$coefficients[, g, ], truth = d$beta[g, ]))
  }

  expect_true(is.list(model$priors))
  expect_equal(length(model$priors), 4)
  expect_equal(names(model$priors), c("coefficient.prior",
    "coefficient.mean.hyperprior", "coefficient.variance.hyperprior",
    "residual.precision.prior"))
})

test_that("NestedRegression works with a fixed prior", {
  d <- SimulateNestedRegressionData()
  model <- NestedRegression(
      suf = d$suf,
      coefficient.prior = MvnPrior(mean = c(1, 2, 3, 4),
                                   variance = diag(c(16, 4, 9, 1))),
      coefficient.mean.hyperprior = FALSE,
      coefficient.variance.hyperprior = FALSE,
      niter = 100,
      seed = seed)
  expect_true(all(model$prior.mean[, 1] == 1))
  expect_true(all(model$prior.mean[, 2] == 2))
  expect_true(all(model$prior.mean[, 3] == 3))
  expect_true(all(model$prior.mean[, 4] == 4))

  expect_true(all(model$prior.variance[, 1, 1] == 16))
  expect_true(all(model$prior.variance[, 2, 2] == 4))
  expect_true(all(model$prior.variance[, 3, 3] == 9))
  expect_true(all(model$prior.variance[, 4, 4] == 1))
})
