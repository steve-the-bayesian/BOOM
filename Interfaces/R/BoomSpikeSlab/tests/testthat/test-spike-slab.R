# Copyright 2010-2018 Google LLC. All Rights Reserved.
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

n <- 200
p <- 10
ngood = 3
niter = 1000
sigma = 8
x <- cbind(1, matrix(rnorm(n * (p - 1)), nrow = n))
beta <- c(rnorm(ngood, sd = 4), rep(0, p - ngood))
y <- rnorm(n, x %*% beta, sigma)
x <- x[, -1]
## 'draws' is fit using a matrix in the parent frame.
draws <- lm.spike(y ~ x, niter = niter, ping = -1, seed = seed)
draws$x <- x
draws$y <- y
draws$true.beta <- beta
draws$true.sigma <- sigma

## 'model.from.data' is fit using a data frame distributed with base R.
model.from.data <- lm.spike(sr ~ ., niter = 100, ping = -1,
  data = LifeCycleSavings, seed = seed)

test_that("lm.spike returns basic info", {
  expect_true(is.list(draws))
  expect_true("prior" %in% names(draws))
  expect_true("beta" %in% names(draws))
  expect_true("sigma" %in% names(draws))

  sigma.lo <- quantile(draws$sigma, .005)
  sigma.hi <- quantile(draws$sigma, .995)
  expect_true(sigma > sigma.lo & sigma < sigma.hi)

  smry <- summary(draws)
  expect_equal(ncol(smry$coef), 5)
})

test_that("lm.spike works with student errors", {
  y[1] <- 500
  model <- lm.spike(y ~ x, niter = niter, ping = -1,
    error.distribution = "student", seed = seed)
  expect_true("prior" %in% names(model))
  expect_true("beta" %in% names(model))
  expect_true("sigma" %in% names(model))
  expect_true("tail.thickness" %in% names(model))

  expect_true(all(model$sigma > 0))
  expect_true(CheckMcmcVector(model$sigma, truth = sigma))
  expect_true(median(model$tail.thickness) < 7)
  expect_true(all(model$tail.thickness > 0))
  expect_true(CheckMcmcMatrix(model$beta, beta),
    info = McmcMatrixReport(model$beta, beta))
})

TestSpikeSlabPrior <- function() {
  ## TODO(steve):  make a test for SpikeSlabPrior
 }

test_that("predict.lm.spike produces prediction", {
  model <- lm.spike(Income ~ ., data = as.data.frame(state.x77),
    niter = 100, ping = -1)
  pred <- predict(model, newdata = as.data.frame(state.x77))
  expect_true(is.matrix(pred))
  expect_true(nrow(pred) == nrow(state.x77))
})

test_that("lm.spike works with degenerate predictor matrix", {
  ## In this test, a design matrix containing a vector of all 0's is
  ## generated.  We should be able to run this matrix through lm.spike
  ## without crashing.
  x1 <- sample(c("a", "b"), 100, TRUE)
  x1a <- x1 == "a"
  x1b <- x1 == "b"
  y <- rnorm(100)
  df <- data.frame(y, x1a, x1b)
  ss.model <- lm.spike("y ~ x1a * x1b", niter = 100, ping = -1, data = df)
})

test_that("plot functions can be called", {
  ## This function just checks that the generic plot functions can be
  ## called without falling over.
  tmp.file <- tempfile()
  png(tmp.file)
  plot(draws)
  plot(draws, "size")
  plot(draws, "residuals")
  plot(draws, "coefficients")
  plot(draws, "scaled.coefficients")
  dev.off()

  tmp.file <- tempfile()
  png(tmp.file)
  plot(model.from.data)
  plot(model.from.data, "size")
  plot(model.from.data, "residuals")
  plot(model.from.data, "coefficients")
  plot(model.from.data, "scaled.coefficients")
  dev.off()
})
