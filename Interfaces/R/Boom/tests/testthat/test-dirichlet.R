library(Boom)
library(testthat)

context("test-dirichlet")
set.seed(8675309)

test_that("ddirichlet works", {
  nu <- 1:3
  probs <- c(.3, .3, .4)
  density <- ddirichlet(probs, nu)
  expect_that(length(density), equals(1))
  expect_gt(density, 0)

  log.density <- ddirichlet(probs, nu, TRUE)
  expect_equal(length(log.density), 1)
  expect_lt(abs(exp(log.density) - density), 1e-8)

  probs <- rbind(c(.1, .5, .4),
                 c(.3, .3, .4))
  density <- ddirichlet(probs, nu)
  expect_equal(length(density), 2)
  expect_true(all(density > 0))
  log.density <- ddirichlet(probs, nu, TRUE)
  expect_equal(length(log.density), 2)
  expect_true(all(abs(exp(log.density) - density)< 1e-8))

  nu <- rbind(1:3,4:6)
  density <- ddirichlet(probs, nu)
  expect_equal(length(density), 2)
  expect_true(all(density > 0))
  log.density <- ddirichlet(probs, nu, TRUE)
  expect_equal(length(log.density), 2)
  expect_true(all(abs(exp(log.density) - density) < 1e-8))

  ## Check that the density matches what you get by direct
  ## computation.
  probs <- runif(4)
  probs <- probs / sum(probs)
  prior.counts <- 1:4
  density <- ddirichlet(probs, prior.counts)
  direct.density <-
    prod(probs^(prior.counts - 1)) *
        prod(gamma(prior.counts)) /
        gamma(sum(prior.counts))
  expect_equal(density, direct.density)

  density <- ddirichlet(probs, rep(1, 4))
  expect_equal(density, 1 / gamma(4))
})

test_that("rdirichlet",  {
  set.seed(8675309)
  nu <- 2:4
  y <- rdirichlet(20, nu);
  expect_equal(20, nrow(y))
  expect_equal(3, ncol(y))
  expect_true(all(abs(rowSums(y) - rep(1, 20)) < 1e-8))

  y <- rdirichlet(1e+6, nu)
  pi.mean <- nu / sum(nu)
  pi.variance <- nu * (sum(nu) - nu) / (sum(nu)^2 * (sum(nu) + 1))
  standard.errors <- sqrt(pi.variance / nrow(y))
  tstats <- (colMeans(y) - pi.mean) / standard.errors
  expect_true(all(abs(tstats) < 2.5))

  y <- rdirichlet(1, nu)
  expect_equal(1.0, sum(y))
  expect_equal(length(nu), length(y))

  y <- rdirichlet(2, rbind(1:3, 4:6))
  expect_equal(2, nrow(y))
  expect_equal(3, ncol(y))
  expect_true(all(rowSums(y) == rep(1, 2)))
})
