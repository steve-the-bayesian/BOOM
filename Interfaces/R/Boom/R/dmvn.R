dmvn <- function(y, mu, sigma, siginv = NULL, ldsi = NULL, logscale = FALSE) {
  ## Density of the multivariate normal distribution.
  ## Args:
  ##   y: A numeric vector or matrix containing the data whose
  ##     density is desired.
  ##   mu:  A vector.  The mean of the distribution.
  ##   sigma: A symmetric, positive definite matrix.  The variance
  ##     matrix of the distribution.
  ##   siginv: The inverse of sigma, or NULL.  If siginv is non-NULL
  ##     then sigma will not be used.
  ##   ldsi:  The log determinant of siginv, or NULL.
  ##   logscale: If TRUE the density is returned on the log scale.
  ##     Otherwise the density is returned on the probability density
  ##     scale.
  ##
  ## Returns:
  ##   The density of y, or of each row of y.
  if (is.vector(y)) {
    y <- matrix(y, nrow = 1)
  }
  stopifnot(is.numeric(mu),
            length(mu) == ncol(y))

  if (is.null(siginv)) {
    stopifnot(is.matrix(sigma),
              nrow(sigma) == ncol(sigma),
              nrow(sigma) == length(mu))
    R <- chol(sigma)
    ## sigma = t(R) %*% R
    ldsi <- -2 * sum(log(diag(R)))

    R.inv <- t(solve(R))
    ## Each column of Y is y[i, ] - mu
    Y <- t(y) - mu
    ## sigma.inverse = t(R.inv) %*% R.inv
    Y <- R.inv %*% Y
    qforms <- colSums(Y ^2)
  } else {
    stopifnot(is.matrix(siginv),
              nrow(siginv) == ncol(siginv),
              nrow(siginv) == length(mu))
    if (is.null(ldsi)) {
      R <- chol(siginv)
      ldsi <- 2 * sum(log(diag(R)))
    }
    qforms <- mahalanobis(y, center = mu, cov = siginv, inverted = TRUE)
  }
  d <- length(mu)
  log2pi <- log(2 *pi)
  ans <- -(d/2) * log2pi + .5 * ldsi - .5 * qforms
  if (logscale) {
    return(ans)
  } else {
    return(exp(ans))
  }
}
