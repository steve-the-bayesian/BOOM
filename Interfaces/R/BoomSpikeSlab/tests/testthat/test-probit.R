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
require(MASS)
set.seed(31417)

test_that("random seed gives repeatability", {
  ## Ensure that calls with the same random seed return the same
  ## draws.
  n <- 100
  niter <- 1000
  x <- rnorm(n)
  beta0 <- -3
  beta1 <- 1
  probits <- beta0 + beta1*x
  probabilities <- pnorm(probits)
  y <- as.numeric(runif(n) <= probabilities)

  ## Do the check with model selection turned off, signaled by
  ## expected.model.size > number of variables.
  model1 <- probit.spike(y ~ x, niter = niter, expected.model.size = 3,
                         seed = 31417, ping = -1)
  model2 <- probit.spike(y ~ x, niter = niter, expected.model.size = 3,
                         seed = 31417, ping = -1)
  expect_equal(model1$beta, model2$beta)

  ## Now do the check with model selection turned on.
  model1 <- probit.spike(y ~ x, niter = niter, seed = 31417, ping = -1)
  model2 <- probit.spike(y ~ x, niter = niter, seed = 31417, ping = -1)
  expect_equal(model1$beta, model2$beta)
})

test_that("Pima Indians", {
  if (requireNamespace("MASS")) {
    data(Pima.tr, package = "MASS")
    data(Pima.te, package = "MASS")
    pima <- rbind(Pima.tr, Pima.te)
    niter <- 1000
    sample.size <- nrow(pima)
    m <- probit.spike(type == "Yes" ~ ., data = pima, niter = niter,
                      seed = 31417, ping = -1)
    expect_true(!is.null(m$beta))
    expect_true(is.matrix(m$beta))
    expect_equal(dim(m$beta), c(1000, 8))
    expect_equal(length(m$fitted.probabilities), sample.size)
    expect_equal(length(m$fitted.probits), sample.size)
    expect_equal(length(m$deviance.residuals), sample.size)
    expect_equal(length(m$log.likelihood), niter, ping = -1)
    expect_true(!is.null(m$prior))

    m.summary <- summary(m)
    expect_true(!is.null(m.summary$coefficients))
    expect_true(is.matrix(m.summary$predicted.vs.actual))
    expect_equal(ncol(m.summary$predicted.vs.actual), 2)
    expect_equal(nrow(m.summary$predicted.vs.actual), 10)
    expect_equal(length(m.summary$deviance.r2.distribution), 1000)
    expect_equal(length(m.summary$deviance.r2), 1)
    expect_true(is.numeric(m.summary$deviance.r2))

    # The following is always true regardless of the data.
    expect_true(m.summary$max.log.likelihood >= m.summary$mean.log.likelihood)

    # The following is true for this data.
    expect_true(m.summary$max.log.likelihood > m.summary$null.log.likelihood)

    m <- probit.spike(type == "Yes" ~ ., data = Pima.tr, niter = niter,
                      seed = 31417, ping = -1)
    predictions <- predict(m, newdata = Pima.te)
  }
})

test_that("probit works with small number of cases", {
  ## The model should run and give sensible results even if, e.g. there are
  ## no/all successes or there are very few data points.  This code should
  ## produce a warning about the prior.success.probability begin zero.
  x <- matrix(rnorm(45), ncol = 9)
  x <- cbind(1, x)
  y <- rep(FALSE, nrow(x))
  expect_warning(
      m <- probit.spike(y ~ x, niter = 100, seed = 31417),
      "Fudging around the fact that prior.success.probability is zero. Setting it to 0.14286")
})
