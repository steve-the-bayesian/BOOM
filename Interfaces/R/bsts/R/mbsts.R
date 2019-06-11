# Copyright 2019 Steven L. Scott.  All rights reserved.
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

# ===========================================================================
# This is a multivariate version of the univariate bsts function.  For now it
# only supports Gaussian observations.  In the future the plan is to expand to
# the same set of reward distributions supported by bsts.  This function may
# actually merge with bsts in the future.  It is quite similar.
#
# Expected usage 1:
#   Matrix y, Matrix x
#   ss <- AddSharedLocalLevel(list(), y)
#   model <- mbsts(y ~ x, state.specification = ss, niter = 1000)
#
# Expected usage 2:
#   Data frame with columns for time stamp and series identifier.
#
# There are two data formats to support.  Wide and tall.  If you want regressors
# then you'll need tall.  Tall data requires an "id" and a "timestamp" variable.
# The tall format is more general, but might be less convenient sometimes.
#
# With the tall format it becomes clear that there are two kinds of regressors.
# Those specific to a time point, and those specific to a series at a time
# point.  The second form can be obtained from the first by repeating the
# predictors.
#
# There should be a role for shrinkage across series in choosing the prior.
# That comes later.
#
# Args:
#   formula: Either a formula as one would supply to lm(), or a matrix of data.
#     If a matrix, rows represent time, and columns represent time series.  If a
#     matrix is supplied then set data.format to "wide".
#   shared.state.specification: A list of SharedStateSpecification objects
#     defining the components of state that are shared across multiple series.
#   series.state.specification: A list of StateSpecification objects defining
#     the series-specific components of state.  There are two options for how
#     this argument can be formatted.
#     - It can be a single list with StateSpecification objects as elements.  In
#       this case each object will be used to define the series-specific state
#       for each time series in the response.
#     - It can be a list with length matching the number of time series to be
#       modeled.  Each list element is a list of StateSpecification objects
#       defining the series-specific state for the corresponding time series.
#       This option is less convenient, but allows greater control.
#   data: An optional data frame containing the data referenced in 'formula.'
#   timestamps: A vector of timestamps indicating the time of each observation.
#     If "wide" data are used this argument is optional, as the timestamps will
#     be inferred from the rows of the data matrix.
#   series.id: A factor-like object indicating the time series to which each
#     observation belongs.  This is only necessary for data in "long" format.
#   prior: The prior distribution over any regression coefficients or other
#     parameters of the observatino distribution.
#   contrasts: an optional list containing the names of contrast functions to
#     use when converting factors numeric variables in a regression formula.
#     This argument works exactly as it does in 'lm'.  The names of the list
#     elements correspond to factor variables in your model formula.  The list
#     elements themselves are the names of contrast functions (see
#     help(contrast) and the 'contrasts.arg' argument to
#     'model.matrix.default').  This argument is only used if a model formula is
#     specified.  It can usually be ignored even then.
#   na.action: What to do about missing values.  The default is to allow
#     missing responses, but no missing predictors.  Set this to na.omit or
#     na.exclude if you want to omit missing responses altogether, but do so
#     with care, as that will remove observations from the time series.
#   niter: a positive integer giving the desired number of MCMC draws
#   ping: A scalar.  If ping > 0 then the program will print a status message to
#     the screen every 'ping' MCMC iterations.
#   data.format: Indicates whether the data are in wide or long form.  Wide data
#     will be converted to long format using Bsts::WideToLong.
#   seed: The seed for the C++ random number generator.
#   ...: Extra arguments are passed to DefaultMbstsPrior().
mbsts <- function(formula,
                  shared.state.specification,
                  series.state.specification = NULL,
                  data = NULL,
                  timestamps = NULL,
                  series.id = NULL,
                  prior = NULL,  # TODO
                  opts = NULL,
                  contrasts = NULL,
                  na.action = na.pass,
                  niter,
                  ping = niter / 10,
                  data.format = c("long", "wide"),
                  seed = NULL,
                  ...) {
  data.format <- match.arg(data.format)
  check.nonnegative.scalar(niter)
  check.scalar.integer(ping)
  stopifnot(is.null(seed) || length(seed) == 1)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  # The first step is to properly format the data.  We need an object called
  # data.list that contains the following:
  #  - predictors: a matrix of predictors
  #  - response: a vector of responses
  #  - series: a factor indicating the time series each element of 'response'
  #      belongs to.
  #  - timestamp.info: A list created by TimestampInfo
  #
  # The coming if/then block is longer than you'd like, but it needs to take
  # place here so the 'model.matrix' black magic can work.
  has.regression <- is.language(formula)
  if (has.regression) {
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
      stop("bsts does not allow NA's in the predictors, only the responses.")
    }
    response <- model.response(my.model.frame, "any")
    if (data.format == "wide") {
      response.frame <- WideToLong(response)
      if (is.null(timestamps)) {
        timestamps <- response.frame$time
      } else {
        timestamps <- timestamps[response.frame$time]
      }
      series.id <- response.frame$series.id
      expanded.predictors <- predictors[series.id, ]
      response <- response.frame$values
    }
    stopifnot(length(response) == nrow(predictors))
  } else {
    ## If there is no regression component the data could be in either long or
    ## wide format.  Make sure it is long.
    response <- formula

    if (data.format == "wide") {
      # Handle a data frames like a matrix.
      if (is.data.frame(response) && all(sapply(response, is.numeric))) {
        response <- as.matrix(response)
        ## TODO: Ensure we don't lose zoo timestamps here.
      }
      response.frame <- WideToLong(response)
      timestamps <- response.frame$time
      series.id <- response.frame$series
      response <- response.frame$values
    }

    if (is.data.frame(response) || is.matrix(response)) {
      stop("Please change data.format to 'wide' if passing matrix-valued time series")
    }
    
    predictors <- matrix(1, nrow = length(response), ncol = 1)
    colnames(predictors) <- "Intercept"
  }
  series.id <- as.factor(series.id)
  nseries <- length(levels(series.id))
  data.list <- list(
    predictors = predictors,
    response = response,
    series.id = series.id,
    timestamp.info = TimestampInfo(predictors, NULL, timestamps)
  )
  
  #------------------------------------------------------------------------
  # Check the format of the state specification.
  #------------------------------------------------------------------------
  spec <- .CheckMbstsStateSpecification(shared.state.specification,
    series.state.specification, nseries)
  shared.state.specification <- spec[[1]]
  series.state.specification <- spec[[2]]

  #------------------------------------------------------------------------
  # Ensure the prior has the proper format.
  #------------------------------------------------------------------------
  if (is.null(prior)) {
    prior <- .DefaultMbstsPrior(predictors, response, series.id)
  }
  # The prior for the observation model is a list of spike adn slab priors.
  stopifnot(is.list(prior), length(prior) == nseries,
    all(sapply(prior, inherits, "SpikeSlabPrior")))

  #------------------------------------------------------------------------
  # Check that options is either NULL or a list.  A bit of faith is needed that
  # the list is formatted correctly.
  # ------------------------------------------------------------------------
  if (!is.null(opts)) {
    stopifnot(is.list(opts))
  }
  
  #------------------------------------------------------------------------  
  # Check that all the scalars are actually scalars.
  #------------------------------------------------------------------------  
  if (!is.null(seed)) {
    seed <- as.integer(seed)
    stopifnot(length(seed) == 1)
  }
  check.scalar.integer(niter)
  check.scalar.integer(ping)

  #------------------------------------------------------------------------  
  # Do the work!
  #------------------------------------------------------------------------
  ans <- .Call("analysis_common_r_fit_multivariate_bsts_model_",
    data.list,
    shared.state.specification,
    series.state.specification,
    prior,
    opts, 
    as.integer(niter),
    as.integer(ping),
    seed)
  ans$has.regression <- has.regression
  ans$shared.state.specification <- shared.state.specification
  ans$series.state.specification <- series.state.specification
  ans$prior <- prior
  ans$niter <- niter
  ans$timestamp.info <- data.list$timestamp.info
  ans$series.id <- series.id
  ans$original.series <- response
  ans$predictors <- predictors
  
  #------------------------------------------------------------------------  
  # Final formatting.
  #------------------------------------------------------------------------  
  # Set dimnames.
  series.names <- as.character(levels(series.id))
  predictor.names <- colnames(predictors)
  state.model.names <- sapply(shared.state.specification,
    function(x) x$name)
  ans$nseries <- length(series.names)

  
  dimnames(ans$regression.coefficients) <- list(
    NULL, series.names, predictor.names)
  dimnames(ans$shared.state.contributions) <- list(
    NULL, state.model.names, series.names, NULL)
  
  class(ans) <- "mbsts"
  return(ans)
}

.DefaultMbstsPrior <- function(predictors, response, series) {
  ## Set a default prior on each of the regression models in the mbsts
  ## observation equation.
  ans <- list()
  for (s in sort(unique(series))) {
    index <- series == s
    data.list <- list(predictors = predictors[index, , drop = FALSE],
      response = response[index])
    ans[[s]] <- .SetDefaultPrior(data.list, family = "gaussian")
  }
  return(ans)
}

.CheckMbstsStateSpecification <- function(shared.state.specification,
                                          series.state.specification,
                                          nseries) {
  # Check that the shared- and series-specific state specifictions are filled
  # with legal values.
  #
  # Args:
  #   shared.state.specification: A list of SharedStateSpecification objects
  #     defining the components of state that are shared across multiple series.
  #   series.state.specification: A list of StateSpecification objects defining
  #     the series-specific components of state.  There are two options for how
  #     this argument can be formatted.
  #     - It can be a single list with StateSpecification objects as elements.  In
  #       this case each object will be used to define the series-specific state
  #       for each time series in the response.
  #     - It can be a list with length matching the number of time series to be
  #       modeled.  Each list element is a list of StateSpecification objects
  #       defining the series-specific state for the corresponding time series.
  #       This option is less convenient, but allows greater control.
  #   nseries: The number of time series.
  #
  # Returns:
  #   A 2-item list containing (1) the shared.state.specification, and (2) the
  #   series.state.specification, possibly after expanding it.
  stopifnot(is.list(shared.state.specification),
    all(sapply(shared.state.specification, inherits, "SharedStateModel")))

  # The series.state.specification might be an empty list.
  stopifnot(is.null(series.state.specification)
    || is.list(series.state.specification))
  # If series state specification is not NULL it is either a list of state
  # specificiations to be repeated for each time series, or it is a list of such
  # specifications.
  if (!is.null(series.state.specification) &&
        all(sapply(series.state.specification, inherits, "StateModel"))) {
    series.state.specification <- RepList(series.state.specification, nseries)
  }
  if (!is.null(series.state.specification)) {
    stopifnot(is.list(series.state.specification),
      length(series.state.specification) == nseries)
    for (i in 1:nseries) {
      stopifnot(is.null(series.state.specification[[i]]) ||
                          is.list(series.state.specification[[i]]),
        all(sapply(series.state.specification[[i]], inherits, "StateModel")))
    }
  }
  return(list(shared.state.specification, series.state.specification))
}

