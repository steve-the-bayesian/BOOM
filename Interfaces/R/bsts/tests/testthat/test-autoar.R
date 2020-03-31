library(bsts)
library(testthat)

test_that("AutoAr does not crash on minimal data.", {
  y <- rnorm(4)
  ss <- AddAutoAr(list(), y=y, lags = 4)

  for (i in 1:20) {
    model <- bsts(y, ss, niter = 100, ping = -1)
  }
  expect_true(inherits(model, "bsts"))
})
