rinvgamma <- function(n, shape, rate) {
  ## Returns n draws from the inverse gamma distribution, parameterized so that
  ## if theta ~ InverseGamma(shape, rate) then 1 / theta ~ Gamma(shape, rate),
  ## meaning that 1/theta has mean shape / rate and variance shape / rate^2.
  ##
  ## Args:
  ##   n:  The desired number of draws.
  ##   shape:  The shape parameter.
  ##   rate: The rate parameter.  NOTE: The term 'rate' is used to match the
  ##     corresponding parameter in 'rgamma.' Much of the rest of the world
  ##     calls this parameter the 'scale' parameter.
  ##
  ## Returns:
  ##   A vector of length n containing the draws.
  return(1.0 / rgamma(n, shape, rate))
}

dinvgamma <- function(x, shape, rate, logscale = FALSE) {
  ## Density of the inverse gamma distribution, parameterized so that if theta ~
  ## InverseGamma(shape, rate) then E(1/theta) = shape/rate and Var(1/theta) =
  ## shape/rate^2.
  ##
  ## Args:
  ##   x:  A vector of deviates where the density is to be evaluated.
  ##   shape:  Shape parameter.
  ##   rate:  Rate parameter.  NOTE: The term 'rate' is used to match the
  ##     corresponding parameter in 'rgamma.' Much of the rest of the world
  ##     calls this parameter the 'scale' parameter.
  ##   logscale: Logical.  If TRUE then the density is returned on the log
  ##     scale.  If FALSE the density is returned on the probability scale.
  ##
  ## Returns:
  ##   A vector with length matching x containing the density values.
  ans <- dgamma(1.0 / x, shape, rate, log = logscale)
  if (logscale) {
    ans <- ans - 2 * log(x)
  } else {
    ans <- ans / x^2
  }
  return(ans)
}

pinvgamma <- function(x, shape, rate, lower.tail = TRUE, logscale = FALSE) {
  ## Density of the inverse gamma distribution, parameterized so that if theta ~
  ## InverseGamma(shape, rate) then E(1/theta) = shape/rate and Var(1/theta) =
  ## shape/rate^2.
  ##
  ## No Jacobian is needed for the CDF.  If X is inverse gamma, then 1/X is
  ## gamma.  Thus Pr(X < x) = Pr(1/x < 1/X) = 1 - pgamma(1/x).  Simply call
  ## pgamma(1/x) and negate the lower.tail argument.
  ##
  ## Args:
  ##   x:  A vector of deviates where the density is to be evaluated.
  ##   shape:  Shape parameter.
  ##   rate:  Rate parameter.  NOTE: The term 'rate' is used to match the
  ##     corresponding parameter in 'rgamma.' Much of the rest of the world
  ##     calls this parameter the 'scale' parameter.
  ##   lower.tail: Logical.  If TRUE then the probability to the left of x is
  ##     returned.  If FALSE then the probability to the right of x is returned.
  ##   logscale: Logical.  If TRUE then the density is returned on the log
  ##     scale.  If FALSE the density is returned on the probability scale.
  ##
  ## Returns:
  ##   A vector with length matching x containin the density values.
  return(pgamma(1/x, shape, rate, lower.tail = !lower.tail,
                log.p = logscale))
}

qinvgamma <- function(p, shape, rate, lower.tail = TRUE, logscale = FALSE) {
  ## Args:
  ##   p: A vector of CDF values (if lower.tail is TRUE) or survivor function
  ##     values (if lower.tail = FALSE) from the inverse gamma distribution.
  ##   shape:  Shape parameter.
  ##   rate:  Rate parameter.  NOTE: The term 'rate' is used to match the
  ##     corresponding parameter in 'rgamma.' Much of the rest of the world
  ##     calls this parameter the 'scale' parameter.
  ##   lower.tail: Logical.  If TRUE then deviates are constructed using
  ##     probability counting from zero (i.e. using the CDF).  If FALSE then
  ##     deviates are constructed using probability counting from infinity
  ##     (i.e. the survivor function).
  ##   logscale: Logical.  If TRUE then the density is returned on the log
  ##     scale.  If FALSE the density is returned on the probability scale.
  ##
  ## Returns:
  ##   A vector of deviates corresponding to the probabilities in 'p'.
  return(1.0 / qgamma(p, shape, rate, lower.tail = !lower.tail,
                      log.p = logscale))
}
