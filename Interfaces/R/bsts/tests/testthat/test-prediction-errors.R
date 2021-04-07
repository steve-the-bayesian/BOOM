library(bsts)
library(testthat)

test_that("Scaled prediction errors are reasonable.", {
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  errors <- bsts.prediction.errors(model, burn = 100)
  se <- bsts.prediction.errors(model, burn = 100, standardize = TRUE)

  ## The scaled and unscaled errors should be the same size.
  expect_equal(dim(se[[1]]), dim(errors[[1]]))

  ## The errors should be highly but not perfectly correlated.
  expect_gt(cor(se[[1]][30, ], errors[[1]][30, ]), .8)
  expect_lte(cor(se[[1]][30, ], errors[[1]][30, ]), 1.0)
})

test_that("Prediction errors work for student family", {
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500, family="student")
  errors <- bsts.prediction.errors(model, cutpoints = c(80, 120))
})
