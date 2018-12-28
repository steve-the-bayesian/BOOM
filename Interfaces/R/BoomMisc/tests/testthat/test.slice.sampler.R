library(BoomMisc)
library(testthat)

context("Slice sampler")

set.seed(8675309)
n <- 300
y <- rgamma(n, 3, 7)

GammaSuf <- function(y) {
  ## Compute sufficient statistics for the gamma distribution.
  return(list(n = length(y),
              sum.y = sum(y),
              sumlog.y = sum(log(y))))
}
suf <- GammaSuf(y)

log.posterior <- function(theta) {
  ## Evaluate the log posterior density for gamma data on the scale log(a/b),
  ## log(a), which allows the inputs to be any real numbers, and substantially
  ## reduces correlation between them.
  ##
  ## The prior is independent standard normal on log(a/b) and log(a).
  a <- exp(theta[2])
  mu <- exp(theta[1])
  b <- a / mu
  ans <- suf$n * a * log(b) - suf$n * lgamma(a) + (a-1) * suf$sumlog.y - b * suf$sum.y
  ans <- ans + dnorm(theta[1], log = TRUE) + dnorm(theta[2], log = TRUE)
}

draws <- slice.sampler(log.posterior, c(0, 0), niter = 1000, ping = 0)

test_that("draws is the right size", {
  expect_true(is.matrix(draws))
  expect_equal(nrow(draws), 1000)
  expect_equal(ncol(draws), 2)
  })

Gammafy <- function(draws) {
  ## Untransform the matrix of draws from log(a/b), log(a) -> (a, b).  This
  ## "Gamma-fies" them in the sense that they're now on the expected scale.
  if (!is.matrix(draws)) {
    draws <- matrix(draws, nrow = 1)
  }
  a <- exp(draws[, 2])
  mu <- exp(draws[, 1])
  b <- a / mu
  return(cbind(a, b))
}

test_that("draws covers true values", {
  expect_true(CheckMcmcMatrix(Gammafy(draws), c(3, 7)))
})

