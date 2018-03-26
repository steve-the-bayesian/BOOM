rmvn <- function(n = 1, mu, sigma = diag(rep(1, length(mu)))) {
  ## Draw from the multivariate normal distribution with mean
  ## vector mu and covariance matrix sigma.
  ##
  ## Args:
  ##   n:  The number of observations to draw.
  ##   mu:  The mean of the distribution.
  ##   sigma:  The variance of the distribution.
  ##
  ## Returns:
  ##   If n == 1 the return value is a vector.  Otherwise it is a
  ##   matrix where each row is a draw.
  stopifnot(is.matrix(sigma),
            nrow(sigma) == ncol(sigma))
  stopifnot(is.numeric(mu),
            length(mu) == nrow(sigma))
  if (sum(eigen(sigma)$values > 0) != length(mu)) {
    warning("covariance matrix is not positive definite!")
  }
  L <- t(chol(sigma))
  p <- length(mu)
  if (n == 1) {
    return(as.numeric(L %*% matrix(rnorm(p), ncol = 1) + mu))
  } else {
    return( t(L %*% matrix(rnorm(p * n), ncol = n) + as.numeric(mu)))
  }
}
