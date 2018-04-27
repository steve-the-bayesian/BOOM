library(BoomSpikeSlab)
library(testthat)
set.seed(2017)
nobs <- 10
p <- 3
ngood <- 2
niter <- 100
sigma <- .8

x <- cbind(1, matrix(rnorm(nobs * (p - 1)), nrow = nobs))
beta <- c(rnorm(ngood), rep(0, p - ngood))
y <- rnorm(nobs, x %*% beta, sigma)
x <- x[, -1]
m1 <- lm.spike(y ~ x, niter = niter, seed = 2017)
m2 <- lm.spike(y ~ x, niter = niter, seed = 2017)

test_that("Models with same seed are identical", {
  expect_that(m1$sigma, equals(m2$sigma))
  expect_that(m1$beta, equals(m2$beta))
})
