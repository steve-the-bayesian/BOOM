LogIntegratedGaussianLikelihood <- function(suf, prior) {
  ## Return the log of the integrated Gaussian likelihood with respect to the
  ## normal inverse gamma prior 'prior'.
  ##
  ## Args:
  ##   suf:  An object of class GaussianSuf.
  ##   prior:  An object of class NormalInverseGammaPrior.
  ##
  ## Returns:
  ##   A scalar, giving the log of the integrated Gaussian likelihood.
  n <- suf$n
  if (n > 0) {
    ybar <- suf$sum / n
  } else {
    ybar <- 0
  }
  if (n > 1) {
    sample.variance <- (suf$sumsq - n * ybar^2) / (n - 1)
  } else {
    sample.variance <- 0
  }
  kappa <- prior$mu.guess.weight
  mu0 <- prior$mu.guess
  df <- prior$sigma.prior$prior.df
  ss <- prior$sigma.prior$prior.guess^2 * df
  
  posterior.mean <- (n * ybar + kappa * mu0) / (n + kappa)
  DF <- df + n
  SS <- ss + (n - 1) * sample.variance + n * (ybar - posterior.mean)^2 +
    kappa * (mu0 - posterior.mean)^2
  return( - .5 * n * log(2 * pi) + .5 * log(kappa / (n + kappa)) +
            lgamma(DF/2) - lgamma(df / 2) +
            .5 * df * log(ss / 2) - .5 * DF * log(SS / 2))
}


