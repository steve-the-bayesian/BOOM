set.seed(8675309)

library(bsts)
library(testthat)
SimulateDirmData <- function(observation.sd = 1, trend.sd = .1,
                             time.dimension = 100, nobs.per.period = 3,
                             xdim = 4) {
  trend <- cumsum(rnorm(time.dimension, 0, trend.sd))
  total.sample.size <- nobs.per.period * time.dimension
  predictors <- matrix(rnorm(total.sample.size * xdim),
    nrow = total.sample.size)
  coefficients <- rnorm(xdim)
  expanded.trend <- rep(trend, each = nobs.per.period)
  response <- expanded.trend + predictors %*% coefficients + rnorm(
    total.sample.size, 0, observation.sd)
  timestamps <- seq.Date(from = as.Date("2008-01-01"), len = time.dimension, by = "day")
  extended.timestamps <- rep(timestamps, each = nobs.per.period)
  return(list(response = response,
    predictors = predictors,
    timestamps = extended.timestamps,
    trend = trend,
    coefficients = coefficients))
}


data <- SimulateDirmData()
ss <- AddLocalLevel(list(), data$response,
  sigma.prior = SdPrior(sigma.guess = 0.1, sample.size = 1))

model <- dirm(data$response ~ data$predictors, ss, niter = 50,
  timestamps = data$timestamps, seed = 8675309, expected.model.size = 20)
model2 <- dirm(response ~ predictors, ss, niter = 50, data = data,
  timestamps = data$timestamps, seed = 8675309, expected.model.size = 20)

test_that("Models are identical", {
  expect_that(model, is_a("DynamicIntercept"))
  expect_that(model$coefficients, is_a("matrix"))
  expect_true(all(abs(model$coefficients - model2$coefficients) < 1e-8))
  expect_true(all(abs(model$sigma.obs - model2$sigma.obs) < 1e-8))
})
