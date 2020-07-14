library(bsts)
library(testthat)

test_that("AddAr produces nonzero coefficients", {
  sample.size <- 100
  residual.sd <- .001
  # Actual values of the AR coefficients
  true.phi <- c(-.7, .3, .15)
  ar <- arima.sim(model = list(ar = true.phi), n = sample.size, sd = 3)
  ## Layer some noise on top of the AR process.
  y <- ar + rnorm(sample.size, 0, residual.sd)
  ss <- AddAr(list(), lags = 3, sigma.prior = SdPrior(3.0, 1.0))
  # Fit the model with knowledge with residual.sd essentially fixed at the true
  # value.
  model <- bsts(y,
    state.specification = ss,
    niter = 10,
    prior = SdPrior(residual.sd, 100000))

  expect_true(any(model$AR3.coefficients != 0))
})
