# Copyright 2011 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

AddSemilocalLinearTrend <- function (state.specification = list(),
                                     y = NULL,
                                     level.sigma.prior = NULL,
                                     slope.mean.prior = NULL,
                                     slope.ar1.prior = NULL,
                                     slope.sigma.prior = NULL,
                                     initial.level.prior = NULL,
                                     initial.slope.prior = NULL,
                                     sdy = NULL,
                                     initial.y = NULL) {

  ## Adds a semi-local linear trend component to
  ## state.specification.  A semi-local linear trend model is a
  ## local linear trend where the slope follows a mean reverting AR1
  ## process instead of a random walk.  There are default values for
  ## most parameters, but most of them depend on sd(y), so in order to
  ## avoid computing sd(y) multiple times the defaults have been moved
  ## to the function body.  It is expected that most arguments will be
  ## missing.
  ##
  ## Args:
  ##   state.specification: A list of state components.  If omitted,
  ##     an empty list is assumed.
  ##   y:  A numeric vector.  The time series to be modeled.
  ##   level.sigma.prior: An object created by SdPrior.  The prior
  ##     distribution for the standard deviation of the increments in
  ##     the level component of state.
  ##   slope.mean.prior: An object created by NormalPrior.  The prior
  ##     distribution for the mean of the AR1 process for the slope
  ##     component of state.
  ##   slope.ar1.prior: An object created by Ar1CoefficientPrior.  The
  ##     prior distribution for the ar1 coefficient in the slope
  ##     component of state.
  ##   slope.sigma.prior: An object created by SdPrior.  The prior
  ##     distribution for the standard deviation of the increments in
  ##     the slope component of state.
  ##   initial.level.prior: An object created by NormalPrior.  The
  ##     prior distribution for the level component of state at the
  ##     time of the first observation.
  ##   initial.slope.prior: An object created by NormalPrior.  The
  ##     prior distribution for the slope component of state at the
  ##     time of the first observation.
  ##   sdy: The standard deviation of y.  This can be ignored if y is
  ##     provided, or if all the required prior distributions are
  ##     supplied directly.
  ##   initial.y: The initial value of y.  This can be omitted if y is
  ##     provided.
  ##
  ## Returns:
  ##   state.specification after appending the necessary information
  ##   to define a semi-local linear trend model
  #
  stopifnot(is.list(state.specification))
  if (!is.null(y)) {
    stopifnot(is.numeric(y))
    if (is.null(sdy)) sdy <- sd(y, na.rm = TRUE)
    if (is.null(initial.y)) initial.y <- y[1]
  }

  if (is.null(level.sigma.prior)) {
    ## The prior distribution says that level.sigma is small, and can be no
    ## larger than the sample standard deviation of the time series
    ## being modeled.
    stopifnot(is.numeric(sdy), length(sdy) == 1, sdy > 0)
    level.sigma.prior <- SdPrior(.01 * sdy, upper.limit = sdy)
  }

  if (is.null(slope.sigma.prior)) {
    ## The prior distribution says that slope.sigma is small, and can
    ## be no larger than the sample standard deviation of the time
    ## series being modeled.
    stopifnot(is.numeric(sdy), length(sdy) == 1, sdy > 0)
    slope.sigma.prior <- SdPrior(.01 * sdy, upper.limit = sdy)
  }

  if (is.null(slope.mean.prior)) {
    stopifnot(is.numeric(sdy), length(sdy) == 1, sdy > 0)
    slope.mean.prior <- NormalPrior(0, sdy)
  }

  if (is.null(slope.ar1.prior)) {
    slope.ar1.prior <- Ar1CoefficientPrior()
  }

  if (is.null(initial.level.prior)) {
    stopifnot(is.numeric(initial.y), length(initial.y) == 1)
    stopifnot(is.numeric(sdy), length(sdy) == 1, sdy > 0)
    initial.level.prior <- NormalPrior(initial.y, sdy)
  }

  if (is.null(initial.slope.prior)) {
    stopifnot(is.numeric(sdy), length(sdy) == 1, sdy > 0)
    initial.slope.prior <- NormalPrior(0, sdy)
  }

  spec <- list(name = "trend",
               level.sigma.prior = level.sigma.prior,
               slope.sigma.prior = slope.sigma.prior,
               slope.mean.prior = slope.mean.prior,
               slope.ar1.prior = slope.ar1.prior,
               initial.level.prior = initial.level.prior,
               initial.slope.prior = initial.slope.prior,
               size = 3)
  class(spec) <- c("SemilocalLinearTrend", "StateModel")

  state.specification[[length(state.specification) + 1]] <- spec
  return(state.specification)
}
