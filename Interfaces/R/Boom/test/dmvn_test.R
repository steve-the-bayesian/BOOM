Testdmvn <- function() {
  y <- rnorm(4)
  density <- dmvn(y, rep(0, 4), diag(rep(1, 4)), log = TRUE)
  density.direct <- sum(dnorm(y, log = TRUE))
  checkEquals(density.direct, density)

  Sigma <- crossprod(matrix(rnorm(16), ncol = 4))
  mu <- rnorm(4)
  y <- as.numeric(t(chol(Sigma)) %*% y + mu)

  density <- dmvn(y, mu, Sigma, log = TRUE)
  p <- length(y)
  qform <- mahalanobis(y, mu, Sigma, inverted = FALSE)

  density.direct <- log(1 / sqrt(2 * pi)^p) - .5 * log(det(Sigma)) - .5 * qform
  checkEquals(density.direct, density)
}
