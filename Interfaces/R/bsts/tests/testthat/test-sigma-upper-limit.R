library(bsts)
library(testthat)

test_that("sigma.upper.limit can be specified at the main bsts call.", {
  y <- rnorm(4)
  x <- rnorm(4)
  ss <- AddLocalLevel(list(), y)

  warning("enable the test for 'sigma.upper.limit'")

  ## for (i in 1:20) {
  ##   model <- bsts(y, ss, niter = 100, ping = -1, sigma.upper.limit = 10)
  ## }
  ## expect_true(inherits(model, "bsts"))

  ## for(i in 1:20) {
  ##   model <- bsts(y ~ x, ss, niter = 10, ping = -1, sigma.upper.limit = 10)
  ##   }
})
