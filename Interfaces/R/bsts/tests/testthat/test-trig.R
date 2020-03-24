library(testthat)
library(bsts)
seed <- 8675309
set.seed(seed)

test_that("Multiple frequencies can be present.", {
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddTrig(ss, y, period = 20, frequencies = 1)
  ss <- AddTrig(ss, y, period = 41, frequencies = 1)
  model <- bsts(y, state.specification = ss, niter = 500)
  expect_equal(500, length(model$trig.coefficient.sd.20))
  expect_equal(500, length(model$trig.coefficient.sd.41))
  expect_equal(dimnames(model$state.contributions)$component,
    c("trend", "trig.20", "trig.41"))
})
