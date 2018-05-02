CheckMcmcMatrix <- function(draws, truth, confidence = .95,
                            control.multiple.comparisons = TRUE,
                            burn = 0) {
  ## A utility for unit testing an MCMC algorithm.  Check that an MCMC ensemble
  ## contains the 'true' values used to simulate the fake data driving the MCMC
  ## test.
  ##
  ## Args:
  ##   draws: A matrix of MCMC draws.  Each row is a draw.  Each column is a
  ##     variable in the distribution being sampled by the MCMC.
  ##   truth: The true values used to verify the draws.  A vector of the same
  ##     dimension as ncol(draws).
  ##   confidence: The probability content of the central interval for each
  ##     variable.
  ##   control.multiple.comparisons: If FALSE then the check will fail if any of
  ##     the true values falls outside the corresponding central interval from
  ##     the MCMC sample.  If TRUE then the check will pass as long the fraction
  ##     of intervals covering the true values exceeds a lower bound.  The lower
  ##     bound is the lower limit of the binoimal confidence interval for the
  ##     coverage rate, assuming a true coverage rate of 'confidence'.
  ##
  ## Returns:
  ##   TRUE if the check passes.  FALSE otherwise.
  stopifnot(is.matrix(draws), length(truth) == ncol(draws))
  if (burn > 0) {
    draws <- draws[-(1:burn), , drop = FALSE]
  }
  stopifnot(nrow(draws) > 0)
  alpha <- 1 - confidence
  intervals <- apply(draws, 2, quantile, c(alpha / 2, 1 - (alpha / 2)))
  ## intervals is a 2-row matrix.

  inside <- truth >= intervals[1, ] & truth <= intervals[2, ]
  if (control.multiple.comparisons) {
    fraction.inside <- sum(inside) / length(inside)
    binomial.se <- sqrt(confidence * (1 - confidence) / ncol(draws))
    lower.limit <- confidence - 2 * binomial.se
    return(fraction.inside >= lower.limit)
  } else {
    return(all(inside))
  }
}

McmcMatrixReport <- function(draws, truth, confidence = .95, burn = 0) {
  ## A report that can be put in the 'info' field of an expect_that call when
  ## the MCMC matrix checked by CheckMcmcMatrix fails to cover the true values.
  ##
  ## Args:
  ##   draws: A matrix of MCMC draws.  Each row is a draw.  Each column is a
  ##     variable in the distribution being sampled by the MCMC.
  ##   truth: The true values used to verify the draws.  A vector of the same
  ##     dimension as ncol(draws).
  ##   confidence: The probability content of the central interval for each
  ##     variable.
  ##
  ## Returns:
  ##   A textual representation of a three column matrix.  Each row matches a
  ##   variable in draws, and gives the lower and upper bounds for the credible
  ##   interval used to check the values.  The final column lists the true
  ##   values that are supposed to be inside the credible intervals.  The value
  ##   is returned as a character string that is expected to be fed to cat() or
  ##   print() so that it will render correctly in R CMD CHECK output.
  alpha <- 1 - confidence
  if (burn > 0) {
    draws <- draws[-(1:burn), , drop = FALSE]
  }
  stopifnot(nrow(draws) > 0)
  intervals <- t(apply(draws, 2, quantile, c(alpha / 2, (1 - alpha / 2))))
  ans <- cbind(intervals, truth)
  return(paste(capture.output(print(ans)), collapse = "\n"))
}

CheckMcmcVector <- function(draws, truth, confidence = .95, burn = 0) {
  ## A utility for unit testing an MCMC algorithm.  Check that an MCMC ensemble
  ## contains the 'true' value used to simulate the fake data driving the MCMC
  ## test.
  ##
  ## Args:
  ##   draws:  A numeric vector of Monte Carlo draws.
  ##   truth:  A scalar value used to verify the draws.
  ##   confidence: The probability content of the central interval formed from
  ##     the distribution of 'draws'.
  ##
  ## Details:
  ##   A central interval is formed by taking the middle (1 - confidence)
  ##   proportion of the empirical distribution of 'draws'.  If 'truth' is
  ##   contained in this interval the test is a success and TRUE is returned.
  ##   Otherwise the test is a failure, and FALSE is returned.
  ##
  ## Returns:
  ##    TRUE, if the interval described above covers 'truth', and FALSE otherwise.
  stopifnot(is.numeric(draws),
    is.numeric(truth),
    length(truth) == 1)
  if (burn > 0) {
    draws <- draws[-(1:burn)]
  }
  stopifnot(length(draws) > 0)

  alpha <- 1 - confidence
  interval <- quantile(draws, c(alpha / 2, 1 - (alpha/2)))
  return(truth >= interval[1] && truth <= interval[2])
}
