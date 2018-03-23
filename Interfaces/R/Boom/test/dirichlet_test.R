library(RUnit)

TestDDirichlet <- function() {
  nu <- 1:3
  probs <- c(.3, .3, .4)
  density <- ddirichlet(probs, nu)
  checkTrue(length(density) == 1)
  checkTrue(density > 0)
  log.density <- ddirichlet(probs, nu, TRUE)
  checkTrue(length(log.density) == 1)
  checkEqualsNumeric(exp(log.density), density)

  probs <- rbind(c(.1, .5, .4),
                 c(.3, .3, .4))
  density <- ddirichlet(probs, nu)
  checkTrue(length(density) == 2)
  checkTrue(all(density > 0))
  log.density <- ddirichlet(probs, nu, TRUE)
  checkTrue(length(log.density) == 2)
  checkEqualsNumeric(exp(log.density), density)

  nu <- rbind(1:3,4:6)
  density <- ddirichlet(probs, nu)
  checkTrue(length(density) == 2)
  checkTrue(all(density > 0))
  log.density <- ddirichlet(probs, nu, TRUE)
  checkTrue(length(log.density) == 2)
  checkEqualsNumeric(exp(log.density), density)

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
  checkEquals(density, direct.density)

  density <- ddirichlet(probs, rep(1, 4))
  checkEquals(density, 1 / gamma(4))
}

TestRDirichlet <- function() {
  set.seed(8675309)
  nu <- 2:4
  y <- rdirichlet(20, nu);
  checkEquals(20, nrow(y))
  checkEquals(3, ncol(y))
  checkEqualsNumeric(rowSums(y), rep(1, 20))

  y <- rdirichlet(1e+6, nu)
  pi.mean <- nu / sum(nu)
  pi.variance <- nu * (sum(nu) - nu) / (sum(nu)^2 * (sum(nu) + 1))
  standard.errors <- sqrt(pi.variance / nrow(y))
  tstats <- (colMeans(y) - pi.mean) / standard.errors
  checkTrue(all(abs(tstats) < 2.5))

  y <- rdirichlet(1, nu)
  checkEqualsNumeric(1.0, sum(y))
  checkEquals(length(nu), length(y))

  y <- rdirichlet(2, rbind(1:3, 4:6))
  checkEquals(2, nrow(y))
  checkEquals(3, ncol(y))
  checkEqualsNumeric(rowSums(y), rep(1, 2))
}
