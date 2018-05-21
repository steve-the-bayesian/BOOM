library(Boom)
library(testthat)

context("Wishart")

set.seed(8675309)

test_that("Wishart draws", {
  dimension <- 5
  nu <- 18
  niter <- 10000

  direct.draws <- array(dim = c(niter, dimension, dimension))
  draws <- array(dim = c(niter, dimension, dimension))
  z <- array(rnorm(niter * dimension * nu), dim = c(niter, nu, dimension))
  for (i in 1:niter) {
    direct.draws[i,,] <- crossprod(z[i,,])
    draws[i,,] <- rWishart(nu, diag(rep(1, dimension)))
  }
  ## Check that the simulations "from the definition" and "from the
  ## function" have the same distribution.
  for (i in 1:5) {
    for (j in i:5) {
      test.result <- ks.test(draws[, i, j], direct.draws[, i, j])
      expect_true(test.result$p.value > .05 / 15)  ## Bonferroni correction
    }
  }

  ## Invert the direct.draws, and compare them to another set of draws
  ## from the inverse Wishart distribution.
  for (i in 1:niter) {
    direct.draws[i,,] <- solve(direct.draws[i,,])
    draws[i,,] <- rWishart(nu, diag(rep(1, dimension)), TRUE)
  }
  ## Check that the simulations "from the definition" and "from the
  ## function" have the same distribution.
  for (i in 1:5) {
    for (j in i:5) {
      test.result <- ks.test(draws[, i, j], direct.draws[, i, j])
      expect_true(test.result$p.value > .05 / 15)  ## Bonferroni correction
    }
  }
})

test_that("lmgamma", {
  ans <- lmgamma(6.2, 3)
  log.Mgamma <- log(pi^(3/2)) + lgamma(6.2) + lgamma(6.2 - .5) + lgamma(6.2 - 1)
  expect_equal(ans, log.Mgamma)
})

test_that("traceproduct", {
  ## Check that the function returns the same thing as direct
  ## calculation.
  A <- matrix(rnorm(100), ncol = 10)
  B <- matrix(rnorm(100), ncol = 10)

  trAB <- TraceProduct(A, B, FALSE)
  expect_true(is.numeric(trAB))
  expect_true(length(trAB) == 1)

  trAB.direct <- sum(diag(A %*% B))
  expect_equal(trAB, trAB.direct)

  ## Make B symmetic
  B <- B + t(B)
  trAB <- TraceProduct(A, B, TRUE)
  trAB.direct <- sum(diag(A %*% B))
  expect_equal(trAB, trAB.direct)
})

test_that("Wishart mean", {
  dimension <- 4
  nu <- 7
  niter <- 10000

  S <- rWishart(12, diag(rep(1:dimension)))

  draws <- array(dim = c(niter, dimension, dimension))
  for (i in 1:niter) {
    draws[i,,] <- rWishart(nu, S, FALSE);
  }
  means <- apply(draws, c(2, 3), mean)
  standard.errors <- apply(draws, c(2, 3), sd) / sqrt(niter)
  true.mean <- nu * S
  tstats <- (means - true.mean) / standard.errors
  threshold <- abs(qnorm(.05 / (4 * 5 / 2)))
  for (i in 1:dimension) {
    for (j in i:dimension) {
      expect_true(abs(tstats[i, j]) < threshold)
    }
  }

  for (i in 1:niter) {
    draws[i,,] <- rWishart(nu, S, TRUE);
  }
  means <- apply(draws, c(2, 3), mean)
  standard.errors <- apply(draws, c(2, 3), sd) / sqrt(niter)
  true.mean <- solve(S) / (nu - dimension - 1)
  tstats <- (means - true.mean) / standard.errors
  threshold <- abs(qnorm(.05 / (4 * 5 / 2)))
  for (i in 1:dimension) {
    for (j in i:dimension) {
      expect_true(abs(tstats[i, j]) < threshold)
    }
  }
})

test_that("Posterior distribution", {
  ## This test checks the interpretation of the scale.matrix
  ## parameter as the inverse of the sum of squares.
  set.seed(19710114)
  dimension <- 4
  prior.nu <- 16
  sample.size <- 1000
  prior.ss <- diag(rep(1, dimension))
  Sigma <- rWishart(prior.nu, prior.ss, FALSE)
  lower.Sigma <- t(chol(Sigma))
  z <- rmvn(sample.size, rep(0, dimension), Sigma)
  sum.of.squares <- crossprod(z)
  niter <- 1000
  Sigma.draws <- array(dim = c(niter, dimension, dimension))
  inverted.ss <- solve(prior.ss + sum.of.squares)
  for (i in 1:niter) {
    Sigma.draws[i,,] <- rWishart(prior.nu + sample.size,
                                 inverted.ss,
                                 TRUE)
  }
  for (i in 1:dimension) {
    expect_true(CheckMcmcMatrix(Sigma.draws[,i,], Sigma[i, ]),
      info = paste("trouble in Sigma row", i))
  }
})

test_that("dInverseWishart", {
  variance <- matrix(c(8, 6, 7, 5, 3, 01, 9, 8, 6), ncol = 3)
  variance <- crossprod(variance)

  sumsq <- crossprod(matrix(rnorm(9), ncol = 3))
  nu <- 17
  p <- ncol(sumsq)

  density <- dInverseWishart(variance, sumsq, nu, logscale = TRUE)

  density.direct <- .5 * nu * log(det(sumsq)) -
      (nu * p / 2) * log(2) -
      lmgamma(nu/2, p) -
      .5 * (nu + p + 1) * log(det(variance)) -
      .5 * sum(diag(sumsq %*% solve(variance)))

  expect_equal(density.direct, density)
})
