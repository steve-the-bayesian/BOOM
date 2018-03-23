lmgamma <- function(y, dimension) {
  ## Log of the multivariate gamma distribution of order 'dimension'.
  ## This function appears as part of the normalizing constant for the
  ## Wishart distribution.
  stopifnot(is.numeric(dimension),
            length(dimension) == 1,
            dimension == as.integer(dimension),
            dimension >= 1)
  stopifnot(is.numeric(y), length(y) == 1)
  j <- 1:dimension
  normalizing.constant <- (dimension * (dimension - 1) / 4) * log(pi)
  ans <- sum(lgamma(y + (1 - j) / 2))
  return(ans + normalizing.constant)
}
