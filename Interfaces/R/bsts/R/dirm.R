# Copyright 2018 Steven L. Scott. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

dirm <- function(formula,
                 state.specification,
                 data,
                 prior = NULL,
                 contrasts = NULL,
                 na.action = na.pass,
                 niter,
                 ping = niter / 10,
                 model.options = DirmModelOptions(),
                 timestamps = NULL,
                 seed = NULL,
                 ...) {
  ##   formula: A formula describing the regression portion of the relationship
  ##     between y and X.  Missing values are allowed in neither the predictor
  ##     variables nor the response.    
  ##   state.specification: a list with elements created by AddLocalLinearTrend,
  ##     AddSeasonal, and similar functions for adding components of state.
  ##   data: an optional data frame, list or environment (or object coercible by
  ##     ‘as.data.frame’ to a data frame) containing the variables in the model.
  ##     If not found in ‘data’, the variables are taken from
  ##     ‘environment(formula)’, typically the environment from which ‘dirm’ is
  ##     called.
  ##   prior: A SpikeSlabPrior for the regression component of the model.  The
  ##     prior for the time series component of the model is specified during
  ##     the creation of state.specification.  A weak default prior will be used
  ##     if 'prior' is NULL.
  ##   contrasts: an optional list containing the names of contrast functions to
  ##     use when converting factors numeric variables in a regression formula.
  ##     This argument works exactly as it does in 'lm'.  The names of the list
  ##     elements correspond to factor variables in your model formula.  The
  ##     list elements themselves are the names of contrast functions (see
  ##     help(contrast) and the 'contrasts.arg' argument to
  ##     'model.matrix.default').  This argument can usually be ignored.
  ##   na.action: What to do about missing values.  The default is to allow
  ##     missing responses, but no missing predictors.  Set this to na.omit or
  ##     na.exclude if you want to omit missing responses altogether, but do so
  ##     with care, as that will remove observations from the time series.
  ##   niter: a positive integer giving the desired number of MCMC draws
  ##   ping: A scalar.  If ping > 0 then the program will print a status message
  ##     to the screen every 'ping' MCMC iterations.
  ##   model.options: A list produced by DirmModelOptions describing some of the
  ##     more esoteric model options.
  ##   timestamps: The timestamp associated with each value of the response.
  ##     This is most likely of type Date or POSIXt.  It is expected that there
  ##     will be multiple observations per time point (otherwise 'bsts' should
  ##     be used instead of 'dirm'), and thus the 'timestamps' argument will
  ##     contain many duplicate values.
  ##   seed: An integer to use as the C++ random seed.  If NULL then
  ##     the C++ seed will be set using the clock.
  ##   ...:  Extra arguments to be passed to SpikeSlabPrior.
  check.nonnegative.scalar(niter)
  check.scalar.integer(ping)
  stopifnot(is.null(seed) || length(seed) == 1)
  stopifnot(inherits(model.options, "DirmModelOptions"))
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }
  ##----------------------------------------------------------------------
  ## Here begins some black magic to extract the responses and the matrix of
  ## predictors from the model formula and either the 'data' argument or from
  ## objects present in the parent frame at the time of calling.  Most of this
  ## was copied from 'lm'.
  function.call <- match.call()
  my.model.frame <- match.call(expand.dots = FALSE)
  frame.match <- match(c("formula", "data", "na.action"),
    names(my.model.frame), 0L)
  my.model.frame <- my.model.frame[c(1L, frame.match)]
  my.model.frame$drop.unused.levels <- TRUE

  # In an ordinary regression model the default action for NA's is to delete
  # them.  This makes sense in ordinary regression models, but is dangerous in
  # time series, because it artificially shortens the time between two data
  # points.  If the user has not specified an na.action function argument then
  # we should use na.pass as a default, so that NA's are passed through to the
  # underlying C++ code.
  if (! "na.action" %in% names(my.model.frame)) {
    my.model.frame$na.action <- na.pass
  }
  my.model.frame[[1L]] <- as.name("model.frame")
  my.model.frame <- eval(my.model.frame, parent.frame())
  model.terms <- attr(my.model.frame, "terms")
  
  predictors <- model.matrix(model.terms, my.model.frame, contrasts)
  if (any(is.na(predictors))) {
    stop("dirm does not allow NA's in the predictors.")
  }
  
  response <- model.response(my.model.frame, "any")
  ## Check that response and predictors are the right size.  The response
  ## might be a matrix if the model family is logit or Poisson.
  sample.size <- if (is.matrix(response)) nrow(response) else length(response)
  stopifnot(nrow(predictors) == sample.size)
  ## End of black magic to get predictors and response.

  ## Grab the timestamps for the response before passing it to
  ## .FormatBstsDataAndOptions so we can use them later.
  if (missing(data)) {
    ## This should be handled in the argument list by setting a default argument
    ## data = NULL, but doing that messes up the "regression black magic"
    ## section above.
    data <- NULL
  }
  timestamp.info <- .ComputeTimestampInfo(response, data, timestamps)

  data.list <- list(response = as.numeric(response),
    predictors = predictors,
    response.is.observed = !is.na(response),
    timestamp.info = timestamp.info) 

  ##--------------------------------------------------------------------------
  ## Assign a default prior if no prior was supplied.
  if (is.null(prior)) {
    prior <- SpikeSlabPrior(
      x = predictors,
      y = response,
      optional.coefficient.estimate = rep(0, ncol(predictors)),
      sigma.upper.limit = 1.2 * sd(response, na.rm = TRUE),
      ...)
  }
  stopifnot(inherits(prior, "SpikeSlabPriorBase"))
  ## Identify any predictor columns that are all zero, and assign them zero
  ## prior probability of being included in the model.  This must be done after
  ## the prior has been validated.
  predictor.sd <- apply(predictors, 2, sd)
  prior$prior.inclusion.probabilities[predictor.sd <= 0] <- 0
  if (is.null(prior$max.flips)) {
    prior$max.flips <- -1
  }
  ans <- .Call("analysis_common_r_fit_dirm_",
               data.list,
               state.specification,
               prior,
               model.options,
               niter,
               ping,
               seed,
               PACKAGE = "bsts")
  ans$has.regression <- TRUE
  ans$state.specification <- state.specification
  ans$prior <- prior
  ans$timestamp.info <- timestamp.info
  ans$niter <- niter
    if (!is.null(ans$ngood)) {
    ans <- .Truncate(ans)
  }
  ans$original.series <- response

  ##----------------------------------------------------------------------
  ## Add names to state.contributions.
  ## Store the names of each state model in the appropriate dimname
  ## for state.contributions.
  number.of.state.components <- length(state.specification)
  state.names <- character(number.of.state.components)
  for (i in seq_len(number.of.state.components)) {
    state.names[i] <- state.specification[[i]]$name
  }
  dimnames(ans$state.contributions) <- list(
    mcmc.iteration = NULL,
    component = state.names,
    time = NULL)
  ##----------------------------------------------------------------------
  ## Put all the regression junk back in, so things like predict() will work.
  ans$contrasts <- attr(predictors, "contrasts")
  ans$xlevels <- .getXlevels(model.terms, my.model.frame)
  ans$terms <- model.terms
  
  ## Save the predictors, and assign names to the regression coefficients.
  ans$predictors <- predictors
  variable.names <- colnames(predictors)
  if (!is.null(variable.names)) {
    colnames(ans$coefficients) <- variable.names
  }
  class(ans) <- c("DynamicIntercept", "bsts")
  return(ans)
}

DirmModelOptions <- function(timeout.seconds = Inf,
                             high.dimensional.threshold.factor = 1.0) {
  ## Args:
  ##   timeout.seconds: The number of seconds that sampler will be allowed to
  ##     run.  If the timeout is exceeded the returned object will be truncated
  ##     to the final draw that took place before the timeout occurred, as if
  ##     that had been the requested value of 'niter'.  A timeout is reported
  ##     through a warning.
  ##
  ##   high.dimensional.threshold.factor:
  stopifnot(is.numeric(timeout.seconds),
    length(timeout.seconds) == 1,
    timeout.seconds >= 0)
  stopifnot(is.numeric(high.dimensional.threshold.factor),
    length(high.dimensional.threshold.factor) == 1,
    high.dimensional.threshold.factor >= 0)
  ans <- list(timeout.seconds = timeout.seconds,
    high.dimensional.threshold.factor = high.dimensional.threshold.factor)
  class(ans) <- "DirmModelOptions"
  return(ans)
}

.RemoveIntercept <- function(predictors) {
  ## Args:
  ##   predictors:  A matrix of predictor variables.
  ##
  ## Returns:
  ##   predictors, with any intercept terms (columns of all 1's) removed.
  stopifnot(is.matrix(predictors))
  is.intercept <- rep(FALSE, ncol(predictors))
  for (i in 1:ncol(predictors)) {
    if (all(predictors[, i]) == 1) {
      is.intercept[i] <- TRUE
    }
  }
  return(predictors[, !is.intercept])
}
