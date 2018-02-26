# Copyright 2012-2016 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

AddAr <- function(state.specification,
                  y,
                  lags = 1,
                  sigma.prior,
                  initial.state.prior = NULL,
                  sdy) {
  ## Adds an autoregressive process (also known as AR(p)) to the
  ## state.specification.  An AR(p) process assumes
  ##
  ## alpha[t] = phi[1] * alpha[t-1] + ... + phi[p] * alpha[t-p] + N(0, sigma^2).
  ##
  ## The vector of coefficients phi is constrained so that the polynomial
  ##
  ##   1 - phi[1] * z - phi[2] * z^2 - ... -phi[p] * z^p
  ##
  ## has all of its roots outside the unit circle, which is the
  ## necessary and sufficient condition for the process to be
  ## stationary.
  ##
  ## Args:
  ##   state.specification: A list of state components.  If omitted,
  ##     an empty list is assumed.
  ##   y:  A numeric vector.  The time series to be modeled.
  ##   lags:  The number of lags ("p") in the AR process.
  ##   sigma.prior: An object created by SdPrior.  The prior for the
  ##     standard deviation of the process increments.
  ##   initial.state.prior: An object of class MvnPrior describing the
  ##     values of the state at time 0.  This argument can be NULL, in
  ##     which case the stationary distribution of the AR(p) process
  ##     will be used as the initial state distribution.
  ##   sdy: The sample standard deviation of the time series to be
  ##     modeled.
  ##
  ## Details:
  ##   The state for this model at time t is a p-vector with elements
  ##   (alpha[t], alpha[t-1], ..., alpha[t-p+1]).  The observation
  ##   vector is (1, 0, 0, ..., 0).  The transition matrix has
  ##   (phi[1], phi[2], ..., phi[p]) as its first row, and [I_{p-1},
  ##   0] as its subsequent rows.
  ##
  ##   The prior distribution on the AR coefficients ("phi") is
  ##   uniform over the stationary region.
  if (missing(state.specification)) {
    state.specification <- list()
  }
  stopifnot(is.list(state.specification))

  if (!missing(y)) {
    stopifnot(is.numeric(y))
    sdy <- sd(as.numeric(y), na.rm = TRUE)
  }

  if (missing(sigma.prior)) {
    sigma.prior <- SdPrior(.01 * sdy)
  }

  ar.process.spec <- list(name = paste("Ar", lags, sep = ""),
                          lags = as.integer(lags),
                          sigma.prior = sigma.prior,
                          initial.state.prior = initial.state.prior,
                          size = lags)
  class(ar.process.spec) <- c("ArProcess", "StateModel")
  state.specification[[length(state.specification) + 1]] <- ar.process.spec
  return(state.specification)
}


AddAutoAr <- function(state.specification,
                      y,
                      lags = 1,
                      prior = NULL,
                      sdy = NULL,
                      ...) {
  ## An AR(p) process (see AddAr for details) where the AR
  ## coefficients have a spike-and-slab prior.
  ##
  ## Args:
  ##   state.specification:A list of state components.  If omitted, an
  ##     empty list is assumed.
  ##
  ##   y: A numeric vector.  The time series to be modeled.  This can
  ##     be omitted if \code{sdy} is supplied.
  ##
  ##   lags: The maximum number of lags ("p") to be considered in the
  ##     AR(p) process.
  ##
  ##   prior: An object inheriting from 'SpikeSlabArPrior', or NULL.
  ##     If the latter, then a default SpikeSlabArPrior will be
  ##     created.
  ##
  ##   sdy: The sample standard deviation of the time series to be
  ##     modeled.  Used to scale the prior distribution.  This can be
  ##     omitted if 'y' is supplied.
  ##
  ##   ...: Extra arguments passed to 'SpikeSlabArPrior'.
  if (is.null(prior)) {
    if (is.null(sdy)) {
      if (missing(y)) {
        stop("One of 'prior', 'y' or 'sdy' must be passed to AddAutoAr.")
      }
      sdy <- sd(y, na.rm = TRUE)
      if (sdy <= 0) {
        warning("Input data had zero standard deviation.")
        sdy <- 1
      }
    }
    prior <- SpikeSlabArPrior(lags, sdy = sdy, ...)
  }
  stopifnot(inherits(prior, "SpikeSlabArPrior"))

  auto.ar.spec <- list(name = paste("Ar", lags, sep = ""),
                       lags = as.integer(lags),
                       prior = prior,
                       size = lags)
  class(auto.ar.spec) <- c("AutoAr", "ArProcess", "StateModel")
  state.specification[[length(state.specification) + 1]] <- auto.ar.spec
  return(state.specification)
}

GeometricSequence <- function(length, initial.value = 1, discount.factor = .5) {
  ## Produce a geometric sequence a * (b^0 + b^1 + ... + b^length-1).
  ##
  ## Args:
  ##   length:  The length of the desired sequence.
  ##   initial.value:  The first term in the sequence.
  ##   discount.factor: The ratio between consecutive terms in the
  ##     sequence.
  ##
  ## Returns:
  ##   A vector containing the desired sequence.
  stopifnot(is.numeric(length),
            length(length) == 1,
            length > 0,
            length == as.integer(length))
  stopifnot(is.numeric(initial.value),
            length(initial.value) == 1,
            initial.value != 0)
  stopifnot(is.numeric(discount.factor),
            length(discount.factor) == 1,
            discount.factor != 0)
  return(initial.value * discount.factor^(0:(length-1)))
}

SpikeSlabArPrior <- function(
    lags,
    prior.inclusion.probabilities =
        GeometricSequence(lags, initial.value = .8, discount.factor = .8),
    prior.mean = rep(0, lags),
    prior.sd =
        GeometricSequence(lags, initial.value = .5, discount.factor = .8),
    sdy,
    prior.df = 1,
    expected.r2 = .5,
    sigma.upper.limit = Inf,
    truncate = TRUE) {

  stopifnot(is.numeric(prior.inclusion.probabilities),
            length(prior.inclusion.probabilities) == lags,
            all(prior.inclusion.probabilities >= 0),
            all(prior.inclusion.probabilities <= 1))
  stopifnot(is.numeric(prior.mean),
            length(prior.mean) == lags)
  stopifnot(is.numeric(prior.sd),
            length(prior.sd) == lags,
            all(prior.sd > 0))
  prior <- IndependentSpikeSlabPrior(
      prior.inclusion.probabilities = prior.inclusion.probabilities,
      optional.coefficient.estimate = prior.mean,
      prior.beta.sd = prior.sd,
      number.of.variables = lags,
      sigma.upper.limit = sigma.upper.limit,
      sdy = sdy,
      sdx = rep(1, lags),
      expected.r2 = expected.r2,
      prior.df = prior.df)
  prior$truncate <- truncate

  class(prior) <- c("SpikeSlabArPrior", class(prior))
  return(prior)
}
