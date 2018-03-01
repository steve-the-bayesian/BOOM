ddirichlet <- function(probabilities, nu, logscale = FALSE) {
  ## Evaluates the density of the Dirichlet distribution.
  ## Args:
  ##   probabilities: A discrete probability distribution, or a matrix
  ##     with rows that are discrete probability distributions of the
  ##     same dimension.  Zero probabilities are not allowed.
  ##   nu: Parameters of the Dirichlet distribution.  This can be a
  ##     vector of positive numbers, interpretable as prior counts,
  ##     of length matching the dimension of probabilities.  If
  ##     probabilities is a matrix then nu can also be a matrix of the
  ##     same dimension, in which case each row of nu is used to
  ##     evaluate the corresponding row of probabilities.
  ##   logscale: Logical.  If TRUE then return the log of the
  ##     Dirichlet distribution.  If FALSE return the distribution on
  ##     the absolute scale.
  ##
  ## Returns:
  ##   The value of the Dirichlet density at the specified argument.
  stopifnot(is.numeric(probabilities),
            all(probabilities >= 0))
  if (is.matrix(probabilities)) {
    totals <- rowSums(probabilities)
    stopifnot(all(abs(totals - 1.0) < 1e-10))
  } else {
    stopifnot(abs(sum(probabilities) - 1) < 1e-10)
  }

  stopifnot(is.numeric(nu), all(nu > 0))
  if (is.matrix(nu)) {
    stopifnot(dim(nu) == dim(probabilities))
    ans <- rowSums((nu - 1) * log(probabilities)) +
        rowSums(lgamma(nu)) - lgamma(rowSums(nu))
  } else if (is.matrix(probabilities)) {
    stopifnot(ncol(probabilities) == length(nu))
    ans <- colSums((nu - 1) * t(log(probabilities))) +
        sum(lgamma(nu)) - lgamma(sum(nu))
  } else {
    stopifnot(length(probabilities) == length(nu))
    ans <- sum((nu - 1) * log(probabilities)) +
        sum(lgamma(nu)) - lgamma(sum(nu))
  }
  if (logscale) return(ans)
  return(exp(ans))
}

rdirichlet<-function(n, nu) {
  ## Generate random draws from the dirichlet distribution.
  ## Args:
  ##   n:  Number of desired draws.
  ##   nu: The parameters of the distribution.  A vector of positive
  ##     numbers interpretable as prior counts.  Can also be a matrix
  ##     (with n rows), in which case a separate nu applies to each
  ##     draw.
  ##
  ## Returns:
  ##   If n > 1 the return value is a matrix with n rows and dim(nu)
  ##     columns.
  stopifnot(is.numeric(n),
            length(n) == 1,
            n > 0,
            n == as.integer(n))
  if (is.matrix(nu)) {
    stopifnot(nrow(nu) == n,
              all(nu > 0))
    dimension <- ncol(nu)
    ans <- matrix(rgamma(n * dimension, nu), nrow = n)
  } else {
    stopifnot(is.numeric(nu),
              all(nu > 0))
    dimension <- length(nu)
    ans <- t(matrix(rgamma(n * dimension, nu),
                    nrow = dimension))
  }
  if (n == 1) {
    return(as.numeric(ans) / sum(ans))
  } else {
    return(ans / rowSums(ans))
  }
}
