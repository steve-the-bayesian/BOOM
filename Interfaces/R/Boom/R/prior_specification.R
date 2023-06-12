SdPrior <- function(sigma.guess,
                    sample.size = .01,
                    initial.value = sigma.guess,
                    fixed = FALSE,
                    upper.limit = Inf) {
  ## Generates an object of class SdPrior that can be used as an input
  ## to a Bayesian model for a standard deviation paramter.
  ##
  ## Args:
  ##   sigma.guess: An a priori guess at the value of the standard
  ##     deviation modeled by this distribution.
  ##   sample.size: The number of observations worth of weight to give
  ##     to sigma.guess.
  ##   initial.value: The initial value to use (in an MCMC algorithm)
  ##     for a deviate modeled by this distribution.
  ##   fixed: Should the deviate modeled by this distribution be fixed
  ##     at its initial value?  (Used for debugging by some code.  Not
  ##     universal.)
  ##   upper.limit: The largest acceptable value for the standard
  ##     deviation modeled by this distribution.  May be Inf.
  stopifnot(is.numeric(sigma.guess), length(sigma.guess) == 1, sigma.guess > 0)
  stopifnot(is.numeric(sample.size), length(sample.size) == 1, sample.size > 0)
  stopifnot(is.numeric(initial.value), length(initial.value) == 1,
            initial.value > 0)
  stopifnot(is.logical(fixed), length(fixed) == 1)
  stopifnot(is.numeric(upper.limit), length(upper.limit) == 1)
  ans <- list(prior.guess = sigma.guess,
              prior.df = sample.size,
              initial.value = initial.value,
              fixed = fixed,
              upper.limit = upper.limit)
  class(ans) <- c("SdPrior", "DiffDoubleModel", "DoubleModel", "Prior")
  return(ans)
}

NormalPrior <- function(mu, sigma, initial.value = mu, fixed = FALSE) {
  ## Returns a list with the information needed to specify a Gaussian
  ## prior on a scalar parameter.
  ##
  ## Args:
  ##   mu:  Mean of the distribution.
  ##   sigma:  Standard deviation.
  ##   initial.value: The initial value of the deviate to be modeled
  ##     by this distribution in an MCMC algorithm.
  ##   fixed: Should the deviate modeled by this distribution be fixed
  ##     at its initial value?  (Used for debugging by some code.  Not
  ##     universal.)
  stopifnot(is.numeric(mu), length(mu) == 1)
  stopifnot(is.numeric(sigma), length(sigma) == 1, sigma >= 0)
  stopifnot(is.numeric(initial.value), length(initial.value) == 1)
  stopifnot(is.logical(fixed), length(fixed) == 1)
  ans <- list(mu = mu,
              sigma = sigma,
              initial.value = initial.value,
              fixed = fixed)
  class(ans) <- c("NormalPrior", "DiffDoubleModel", "DoubleModel", "Prior")
  return(ans)
}

Ar1CoefficientPrior <- function(mu = 0,
                                sigma = 1,
                                force.stationary = TRUE,
                                force.positive = FALSE,
                                initial.value = mu) {
  ## Returns a list with the information needed to supply a prior
  ## distribution on an AR1 coefficient.
  ans <- NormalPrior(mu, sigma, initial.value)
  ans$force.stationary <- force.stationary
  ans$force.positive <- force.positive
  class(ans) <- c("Ar1CoefficientPrior", class(ans))
  return(ans)
}

BetaPrior <- function(a = 1, b = 1, mean = NULL, sample.size = NULL,
                      initial.value = NULL) {
  ## Returns an object of class "BetaPrior", which is a list
  ## containing the parameters of a beta distribution.  The prior can
  ## either be given in terms of 'a' and 'b', or it can be given in
  ## terms of mean and sample.size, where mean = a/a+b and sample.size
  ## = a+b.
  if (!is.null(sample.size) && !is.null(mean)) {
    stopifnot(is.numeric(mean) &&
              is.numeric(sample.size) &&
              length(mean) == 1 &&
              length(sample.size) == 1)
    a <- mean * sample.size
    b <- (1 - mean) * sample.size
  }

  stopifnot(is.numeric(a),
            length(a) == 1,
            a > 0)
  stopifnot(is.numeric(b),
            length(b) == 1,
            b > 0)
  if (is.null(initial.value)) {
    initial.value <- a / (a + b)
  }
  stopifnot(is.numeric(initial.value),
            length(initial.value) == 1,
            initial.value > 0,
            initial.value < 1);
  ans <- list(a = a, b = b, initial.value = initial.value)
  class(ans) <- c("BetaPrior", "DiffDoubleModel", "DoubleModel", "Prior")
  return(ans)
}

UniformPrior <- function(lo = 0, hi = 1, initial.value = NULL) {
  ## A uniform prior distribution on the interval [lo, hi].
  ## Args:
  ##   lo:  The lower limit of prior support.
  ##   hi:  The upper limit of prior support.
  ## Returns:
  ##   An object of class UniformPrior.
  stopifnot(is.numeric(lo),
            length(lo) == 1)
  stopifnot(is.numeric(hi),
            length(lo) == 1,
            lo <= hi)
  if (is.null(initial.value)) {
    initial.value <- .5 * (lo + hi)
  }
  stopifnot(is.numeric(initial.value),
            length(initial.value) == 1,
            initial.value >= lo,
            initial.value <= hi)
  ans <- list(lo = lo, hi = hi, initial.value = initial.value)
  class(ans) <- c("UniformPrior", "DiffDoubleModel", "DoubleModel", "Prior")
  return(ans)
}

GammaPrior <- function(a = NULL, b = NULL, prior.mean = NULL,
                       initial.value = NULL) {
  ## Gamma distribution with parameters (a, b), where the mean is a/b
  ## and variance is a/b^2.
  ## Args:
  ##   a:  The shape parameter 'a'.
  ##   b:  The scale parameter 'b'.
  ##   prior.mean: This specifies a/b.  If non-NULL at least one of
  ##     'a' or 'b' must be specified.
  ##   initial.value: The initial.value of the variable to be modeled
  ##     in the MCMC algorithm.
  ## Returns:
  ##   An object of class GammaPrior.
  if (is.null(prior.mean)) {
    stopifnot(!is.null(a))
    stopifnot(is.numeric(a))
  } else {
    stopifnot(is.numeric(prior.mean))
    if (!is.null(a) && is.numeric(a)) {
      b <- a / prior.mean
    } else if (!is.null(b) && is.numeric(b)) {
      a <- b * prior.mean
    }
  }
  stopifnot(is.numeric(a))
  stopifnot(is.numeric(b))
  stopifnot(length(a) == 1)
  stopifnot(length(b) == 1)
  stopifnot(all(a > 0))
  stopifnot(all(b > 0))

  if (is.null(initial.value)) {
    initial.value <- a / b
  }
  ans <- list(a = a, b = b, initial.value = initial.value)
  class(ans) <- c("GammaPrior", "DiffDoubleModel", "DoubleModel", "Prior")
  return(ans)
}

TruncatedGammaPrior <- function(a = NULL, b = NULL, prior.mean = NULL,
                                initial.value = NULL,
                                lower.truncation.point = 0,
                                upper.truncation.point = Inf) {
  ## A Gamma distribution with parameter matching 'GammaPrior' but
  ## with support truncated to between lower.truncation.point and
  ## upper.truncation.point.
  ans <- GammaPrior(a, b, prior.mean, initial.value)
  stopifnot(is.numeric(lower.truncation.point),
            length(lower.truncation.point) == 1)
  stopifnot(is.numeric(upper.truncation.point),
            length(upper.truncation.point) == 1,
            upper.truncation.point > lower.truncation.point)
  ans$lower.truncation.point <- lower.truncation.point
  ans$upper.truncation.point <- upper.truncation.point
  if (!is.null(initial.value) && (
      initial.value < lower.truncation.point ||
          initial.value > upper.truncation.point)) {
    stop("Initial value must be between the lower and upper truncation points.")
  } else if (ans$initial.value < lower.truncation.point
             || ans$initial.value > upper.truncation.point) {
    if (is.finite(upper.truncation.point)) {
      warning("Initial value for TruncatedGammaPrior is outside the ",
              "supported range. Changing it to the midpoint of the ",
              "support region.")
      ans$initial.value <- .5 * (lower.truncation.point +
                                     upper.truncation.point)
    } else {
      warning("Initial value for TruncatedGammaPrior is outside the ",
              "supported range. Placing it above the lower bound.")
      ans$initial.value <- lower.truncation.point + 1
    }
  }
  class(ans) <- c("TruncatedGammaPrior", "DiffDoubleModel",
                  "DoubleModel", "Prior")
  return(ans)
}


LognormalPrior <- function(mu = 0.0, sigma = 1.0, initial.value = NULL) {
  ## A lognormal distribution, where log(y) ~ N(mu, sigma).  The mean
  ## of this distribution is exp(mu + 0.5 * sigma^2), so don't only
  ## focus on the mean parameter here.
  ##
  ## Args:
  ##   mu:  mean of the corresponding normal distribution.
  ##   sigma: standard deviation of the corresponding normal
  ##     distribution.  WARNING: If something looks strange in your
  ##     program, look out for SD != Variance errors.
  stopifnot(is.numeric(mu),
            length(mu) == 1,
            is.finite(mu))
  stopifnot(is.numeric(sigma),
            length(sigma) == 1,
            sigma > 0,
            is.finite(sigma))
  if (is.null(initial.value)) {
    initial.value <- exp(mu + .5 * sigma^2)
  }
  stopifnot(is.numeric(initial.value),
            length(initial.value) == 1,
            initial.value > 0,
            is.finite(initial.value))

  ans <- list(mu = mu, sigma = sigma, initial.value = initial.value)
  class(ans) <- c("LognormalPrior", "DiffDoubleModel", "DoubleModel", "Prior")
  return(ans)
}

MarkovPrior <- function(prior.transition.counts = NULL,
                        prior.initial.state.counts = NULL,
                        state.space.size = NULL,
                        uniform.prior.value = 1) {
  ## Prior distribution for a MarkovModel.
  ## Args:
  ##   prior.transition.counts: A matrix of non-negative numbers
  ##     interpretable as prior counts.  Transitions are from rows to
  ##     columns.
  ##   prior.initial.state.counts: A vector of non-negative numbers
  ##     interpretable as prior counts.
  ##   state.space.size: If both prior.transition.counts and
  ##     prior.initial.state.counts are missing then they will be filled
  ##     with an object of dimension state.space.size where all
  ##     entries are set to uniform.prior.value.
  ##   uniform.prior.value: The default value to use for entries of
  ##     prior.transition.counts and prior.initial.state.counts, when
  ##     they are not supplied by the user.
  if (is.null(state.space.size)) {
    if (is.null(prior.transition.counts) && is.null(prior.transition.counts)) {
      stop("Either 'state.space.size' or one of 'prior.transition.counts' or",
           "'prior.initial.state.counts' must be supplied to MarkovPrior")
    }
    if (!is.null(prior.transition.counts)) {
      state.space.size <- nrow(prior.transition.counts)
    } else {
      state.space.size <- length(prior.initial.state.counts)
    }
  }
  stopifnot(state.space.size > 0)
  stopifnot(uniform.prior.value > 0)

  if (is.null(prior.transition.counts)) {
    prior.transition.counts <- matrix(uniform.prior.value, nrow =
      state.space.size, ncol = state.space.size)
  } else {
    stopifnot(is.matrix(prior.transition.counts))
    stopifnot(nrow(prior.transition.counts) == ncol(prior.transition.counts))
    stopifnot(all(prior.transition.counts >= 0))
    .CheckForPositiveValue <- function(x) { return(any(x > 0)) }
    ## Check that at least one positive value is present in each row.
    stopifnot(all(apply(prior.transition.counts, 1, .CheckForPositiveValue)))

    ## If state.space.size and prior.transition.counts are both
    ## present then issue a warning if they don't match, and use
    ## prior.transition.counts.
    if (state.space.size != nrow(prior.transition.counts)) {
      warning("state.space.size is ", state.space.size,
              ", but nrow(prior.transition.counts) is",
              nrow(prior.transition.counts),
              ".  Changing state.space.size to nrow(prior.transition.counts).")
      state.space.size <- nrow(prior.transition.counts)
    }
  }

  if (is.null(prior.initial.state.counts)) {
    prior.initial.state.counts <- rep(uniform.prior.value, state.space.size)
  } else {
    stopifnot(is.numeric(prior.initial.state.counts))
    ##  Allow a prior that places all its mass on a single point.
    stopifnot(all(prior.initial.state.counts >= 0))
    stopifnot(any(prior.initial.state.counts > 0))
  }

  prior <- list(prior.transition.counts = prior.transition.counts,
                prior.initial.state.counts = prior.initial.state.counts)
  class(prior) <- c("MarkovPrior", "Prior")
  return(prior)
}

DirichletPrior <- function(prior.counts, initial.value = NULL) {
  ## Encodes a Dirichlet priors, the conjugate prior for the
  ## multinomial distribution.
  ## Args:
  ##   prior.counts: A vector of positive numbers with dimension
  ##     matching the probability distribution it is modeling.
  ## Returns:
  ##   An object of class DirichletPrior, which is a list containing
  ##   the prior.counts argument, after some type and sanity checking.
  stopifnot(is.numeric(prior.counts))
  stopifnot(length(prior.counts) > 0)
  stopifnot(all(prior.counts > 0))
  if (is.null(initial.value)) {
    initial.value <- prior.counts / sum(prior.counts)
  }
  ans <- list(prior.counts = prior.counts)
  class(ans) <- c("DirichletPrior", "Prior")
  return(ans)
}

NormalInverseGammaPrior <- function(mu.guess,
                                    mu.guess.weight = .01,
                                    sigma.guess,
                                    sigma.guess.weight = 1,
                                    ...) {
  ## A conjugate prior for the mean and variance of a Gaussian
  ## distribution.
  ## Args:
  ##   mu.guess:  Prior guess at the normal mean parameter.
  ##   mu.guess.weight: Number of observations worth of weight
  ##     assigned to mu.guess.
  ##   sigma.guess: A prior guess at the value of the normal standard
  ##     deviation parameter.
  ##   sigma.guess.weight: Number of observations worth of weight
  ##     assigned to sigma.guess.
  ##   ...: extra parameters passed to SdPrior
  ## Returns:
  ##   An object of class NormalInverseGammaPrior, which contains the
  ##   'mu' arguments and an element of type SdPrior
  stopifnot(is.numeric(mu.guess))
  stopifnot(mu.guess.weight > 0)
  ans <- list(mu.guess = mu.guess,
              mu.guess.weight = mu.guess.weight,
              sigma.prior = SdPrior(
                  sigma.guess = sigma.guess,
                  sample.size = sigma.guess.weight,
                  ...))
  class(ans) <- c("NormalInverseGammaPrior", "Prior")
  return(ans)
}

MvnPrior <- function(mean, variance) {
  ## Encodes the mean and variance of a multivariate normal
  ## distribution for use as a prior distribution.
  ## Args:
  ##   mean:  A numeric vector
  ##   variance:  A symmetric positive definite matrix.
  ## Returns:
  ##   An object of class MvnPrior containing the mean and variance
  ##   arguments, after some sanity checking.
  if (length(variance) == 1) {
    variance <- diag(as.numeric(variance), length(mean), length(mean))
  }
  stopifnot(is.matrix(variance))
  stopifnot(nrow(variance) == ncol(variance))
  stopifnot(nrow(variance) == length(mean))
  ans <- list(mean = mean,
              variance = variance)
  class(ans) <- c("MvnPrior", "Prior")
  return(ans)
}

MvnGivenSigmaMatrixPrior <- function(mean, sample.size) {
  stopifnot(is.numeric(mean))
  stopifnot(is.numeric(sample.size),
            length(sample.size) == 1,
            sample.size > 0)
  ans <- list("mean" = mean, "sample.size" = sample.size)
  class(ans) <- c("MvnGivenSigmaMatrixPrior", "Prior")
  return(ans)
}

InverseWishartPrior <- function(variance.guess,
                                variance.guess.weight) {
  ## Conjugate prior distribution for the variance matrix in a multivariate
  ## normal wtih known mean.

  stopifnot(is.matrix(variance.guess))
  stopifnot(is.numeric(variance.guess.weight),
            length(variance.guess.weight) == 1,
            variance.guess.weight > ncol(variance.guess))
  ans <- list(variance.guess = variance.guess,
              variance.guess.weight = variance.guess.weight)
  class(ans) <- c("InverseWishartPrior", "Prior")
  return(ans)
}

NormalInverseWishartPrior <- function(
    mean.guess,
    mean.guess.weight = .01,
    variance.guess,
    variance.guess.weight = nrow(variance.guess) + 1) {
  ## The conjugate prior distribution for the multivariate normal.  A
  ## multivariate generalization of the NormalInverseGammaPrior
  ## Args:
  ##   mean.guess: A numeric vector that is a guess at the value of
  ##     the multivariate normal mean parameter.
  ##   mean.guess.weight: The number of observations worth of weight
  ##     assigned to mean.guess.
  ##   variance.guess: A symmetric positive definite matrix that is a
  ##     guess at the value of the multivariate normal variance
  ##     parameter.
  ##   variance.guess.weight: The number of observations worth of
  ##     weight assigned to variance.guess.
  stopifnot(is.vector(mean.guess))
  stopifnot(is.matrix(variance.guess))
  stopifnot(all(dim(variance.guess) == length(mean.guess)))
  stopifnot(is.numeric(mean.guess.weight),
            length(mean.guess.weight) == 1,
            mean.guess.weight > 0)
  stopifnot(is.numeric(variance.guess.weight),
            length(variance.guess.weight) == 1,
            variance.guess.weight > 0)

  ans <- list(mean.guess = mean.guess,
              mean.guess.weight = mean.guess.weight,
              variance.guess = variance.guess,
              variance.guess.weight = variance.guess.weight)
  class(ans) <- c("NormalInverseWishartPrior", "Prior")
  return(ans)
}

MvnDiagonalPrior <- function(mean.vector, sd.vector) {
  ## A multivariate normal distribution with a diagonal variance
  ## matrix (i.e. the product of several independent normals).
  ##
  ## Args:
  ##   mean.vector:  The mean of the multivariate normal.
  ##   sd.vector: The standard deviations of the elements of the
  ##     multivariate normal.
  ##
  ## Returns:
  ##   An object of class MvnDiagonalPrior, which is a list containing
  ##   mean.vector and sd.vector.
  stopifnot(is.numeric(mean.vector))
  stopifnot(is.numeric(sd.vector))
  stopifnot(all(sd.vector > 0))
  stopifnot(length(mean.vector) == length(sd.vector))

  ans <- list(mean = mean.vector, sd = sd.vector)
  class(ans) <- c("MvnDiagonalPrior", "Prior")
  return(ans)
}

MvnIndependentSigmaPrior <- function(mvn.prior, sd.prior.list) {
  ## A prior for the parameters of the multivariate normal
  ## distribution that assumes Sigma to be a diagonal matrix with
  ## elements modeled by independent inverse Gamma priors.
  ##
  ## Args:
  ##   mvn.prior: An object of class MvnPrior that is the prior
  ##     distribution of the multivariate normal mean parameter.
  ##   sd.prior.list: A list of SdPrior object modeling the diagonal
  ##     elements of the multivariate normal variance matrix.  The
  ##     off-diagonal elements are assumed to be zero.
  ##
  ## Returns:
  ##   An object of clas MvnIndependentSigmaPrior, which is a list
  ##   containing the function arguments.
  stopifnot(inherits(mvn.prior, "MvnPrior"))
  stopifnot(is.list(sd.prior.list))
  stopifnot(length(sd.prior.list) == length(mvn.prior$mean))
  stopifnot(all(sapply(sd.prior.list, inherits, "SdPrior")))

  ans <- list(mu.prior = mvn.prior,
              sigma.prior = sd.prior.list)
  class(ans) <- c("MvnIndependentSigmaPrior", "Prior")
  return(ans)
}

ScaledMatrixNormalPrior <- function(mean, nu) {
  # A prior for multivariate regression coefficients Beta, such that
  # Beta | Sigma ~ MatrixNormal(mu, X'X / nu, Sigma)
  #
  # Args:
  #   mean:  A matrix of the same size as Beta (ydim * xdim).
  #   nu: A scalar controlling the amount of prior correlation attributable to
  #     rows of beta.  In the context of multivariate regression, rows of beta
  #     correspond ot different Y variables.  A useful value for nu is kappa /
  #     sample.size, where kappa is the number of prior observations worth of
  #     weight you wish to give to 'mean'.
  #
  # Details:
  #   The Sigma parameter is obtained elsewhere.
  stopifnot(is.matrix(mean))
  stopifnot(is.numeric(nu), length(nu) == 1, nu > 0)
  ans <- list("mean" = mean, "nu" = nu)
  class(ans) <- c("ScaledMatrixNormalPrior", "Prior")
  return(ans)
}

MultivariateRegressionConjugatePrior <-function(coefficient.prior,
                                                variance.prior) {
  ## A conjguate prior for a multivariate regression model.
  ##
  ## Args:
  ##   coefficient.prior: An object of class ScaledMatrixNormalPrior
  ##     representing the prior on the regression coefficients.
  ##   variance.prior: An object of class InverseWishartPrior representing the
  ##     prior on the variance matrix.
  ##
  ## Returns:
  ##   A list containing the two arguments, after checking that they are the
  ##   right type.  The return value is marked with a class attribute
  ##   documenting its role.
  stopifnot(inherits(coefficient.prior, "ScaledMatrixNormalPrior"),
    inherits(variance.prior, "InverseWishartPrior"))
  ans <- list(coefficient = coefficient.prior, variance = variance.prior)
  class(ans) <- c("MultivariateRegressionConjugatePrior", "Prior")
  return(ans)
}

RegressionCoefficientConjugatePrior <- function(
    mean,
    sample.size,
    additional.prior.precision = numeric(0),
    diagonal.weight = 0) {
  ## A conditional prior for the coefficients (beta) in a linear regression
  ## model.  The prior is conditional on the residual variance sigma^2, the
  ## sample size n, and the design matrix X.  The prior is
  ##
  ##      beta | sigsq, X ~ N(b, sigsq * (Lambda^{-1} + V))
  ##
  ## where V^{-1} = ((1 - w) * XTX + w * Diag(XTX)) * kappa / n.
  ##
  ## Args:
  ##   mean:  The mean of the prior distribution, denoted 'b' above.
  ##   sample.size: The value denoted 'kappa' above.  This can be
  ##     interpreted as a number of observations worth of weight to be
  ##     assigned to 'b' in the posterior distribution.
  ##   additional.prior.precision: A vector of non-negative numbers
  ##     representing the diagonal matrix Lambda^{-1} above.  Positive
  ##     values for additional.prior.precision will ensure the
  ##     distribution is proper even if the regression model has no
  ##     data.  If all columns of the design matrix have positive
  ##     variance then additional.prior.precision can safely be set to
  ##     zero.  A zero-length numeric vector is a slightly more
  ##     efficient equivalent to a vector of all zeros.
  ##   diagonal.weight: The weight given to the diagonal when XTX is
  ##     averaged with its diagonal.  The purpose of diagonal.weight
  ##     is to keep the prior distribution proper even if X is less
  ##     than full rank.  If the design matrix is full rank then
  ##     diagonal.weight can be set to zero.
  ##
  ## Details:
  ##   The prior distribution also depends on the cross product matrix
  ##   XTX and the sample size n, which are not arguments to this
  ##   function.  It is expected that the underlying C++ code will get
  ##   those quantities elsewhere (presumably from the regression
  ##   modeled by this prior).
  stopifnot(is.numeric(mean))
  stopifnot(is.numeric(sample.size),
            length(sample.size) == 1,
            sample.size > 0)
  stopifnot(is.numeric(additional.prior.precision),
            length(additional.prior.precision) == length(mean)
            || length(additional.prior.precision) == 0)
  if (length(additional.prior.precision) > 0) {
    stopifnot(all(additional.prior.precision >= 0))
  }
  stopifnot(is.numeric(diagonal.weight),
            length(diagonal.weight) == 1,
            diagonal.weight >= 0,
            diagonal.weight <= 1)
  ans <- list(mean = mean,
              sample.size = sample.size,
              additional.prior.precision = additional.prior.precision,
              diagonal.weight = diagonal.weight)
  class(ans) <- c("RegressionCoefficientConjugatePrior")
  return(ans)
}

PointMassPrior <- function(location) {
  ## A prior putting a point mass (probability = 1) on a single scalar
  ## location.
  ## Args:
  ##   location:  The location of the point mass on the real line.
  ans <- list(location = location)
  class(ans) <- c("PointMassPrior", "DiscretePrior", "Prior")
  return(ans)
}

PoissonPrior <- function(mean, lower.limit = 0, upper.limit = Inf) {
  ## Prior over the positive integers based on a Poisson distribution.
  ## Can optionally be truncated to {lower.limit, ..., upper.limit},
  ## including the endpoints.

  ans <- list(mean = mean,
              lower.limit = lower.limit,
              upper.limit = upper.limit)
  class(ans) <- c("PoissonPrior", "DiscretePrior", "Prior")
  return(ans)
}

DiscreteUniformPrior <- function(lower.limit, upper.limit) {
  ## A discrete uniform distribution over the integers in
  ## {lower.limit, ..., upper.limit}.  The end points are included.
  ## Args:
  ##   lower.limit:  The lower limit of the support.
  ##   upper.limit:  The upper limit of the support.
  ans <- list(lower.limit = lower.limit,
              upper.limit = upper.limit)
  class(ans) <- c("DiscreteUniformPrior", "DiscretePrior", "Prior")
  return(ans)
}
