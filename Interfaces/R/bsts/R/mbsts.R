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


# This is a multivariate version of the univariate bsts function.  For now we
# are only supporting Gaussian observations.  In the future the plan is to
# expand to the same set of reward distributions supported by bsts.  This
# function may actually merge with bsts in the future, because it is quite
# similar.
#
# Expected usage:
#   Matrix y, Matrix x
#   ss <- AddSharedLocalLevel(list(), y)
#   model <- mbsts(y ~ x, state.specification = ss, niter = 1000)
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
#   data.format: Indicates whether the data are in wide or long form.
#   seed: The seed for the C++ random number generator.
#   ...: Extra arguments are passed to DefaultMbstsPrior().
mbsts <- function(formula,
                  shared.state.specification,
                  series.state.specification,
                  data,
                  timestamps = NULL,
                  series.id = NULL,
                  prior = DefaultMbstsPrior(),  # TODO
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
  has.regression <- !is.matrix(formula)
  if (has.regression) {
    ## If the model has a regression component then the data must be in long
    ## format.
    if (data.format == "wide") {
      stop("Problems with a regression component require data in long format. ",
        "See help(bsts::ToLong) for help converting your data.")
    }
    
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

    response <- model.response(my.model.frame, "numeric")
    stopifnot(nrow(response) == nrow(predictors))
  } else {
    ## If there is no regression component the data could be in either long or
    ## wide format.  Make sure it is long.
    response <- formula
    if (data.format == "wide") {
      response.frame <- ToLong(response)
      timestamps <- response.frame$time
      series.id <- response.frame$series
      response <- response.frame$values
    }
    predictors <- NULL
  }

  if (is.null(predictors)) {
    predictors <- matrix(1.0, nrow = sample.size, ncol = 1)
  }
  
  if (missing(data)) {
    # This should be handled in the argument list by setting a default argument
    # data = NULL, but doing that messes up the "regression black magic" section
    # above.
    data <- NULL
  }

  data.list <- list(response = response,
    predictors = predictors,
    series.id = series.id,
    timestamp.info = .ComputeTimestampInfo(
      rep(1, length(timestamps)), NULL, timestamps)
  )
  
  if (is.null(prior)) {
    stop("Need to implement a default prior")
    ## We want a vector of SdPrior's for the residual variance.  We want a
    ## shrinkage spike-and-slab prior for the coefficients... Need to work this
    ## out.
  }
  stopifnot(inherits(prior, "mbstsPrior"))

  ans <- .Call("analysis_common_r_fit_multivariate_bsts_model_",
    data.list,
    shared.state.specification,
    series.state.specification,
    prior,
    NULL,  # slot for model.options.
    niter,
    ping,
    seed)

  ### Next do cleanup.
  
}

ToWide <- function(response, series.id, timestamps) {
  ## Convert a multivariate time series in "long" format to "wide" format.
  ##
  ## Args:
  ##   response:  The time series values.
  ##   series.id: A vector of labels of the same length as 'response' indicating
  ##     the time series to wihch each element of 'response' belongs.
  ##   timestamps:  The time period to which each observation belongs.
  ##
  ## Returns:
  ##   A zoo matrix with rows corresponding to time stamps and columns
  ##   corresponding to different time series.  The matrix elements are the
  ##   'response' values.
  ##
  ## Note:
  ##   This could be done with 'reshape'.  I have reworked things by hand in the
  ##   interest of readability.
  stopifnot(length(response) == length(series.id),
    length(response) == length(timestamps))
  unique.times <- sort(unique(timestamps))
  unique.names <- unique(series.id)
  ntimes <- length(unique.times)
  nseries <- length(unique.names)

  ans <- matrix(nrow = ntimes, ncol = nseries)
  if (ntimes == 0 || nseries == 0) {
    return(ans)
  }
  colnames(ans) <- as.character(unique(series.id))

  for (i in 1:ntimes) {
    index <- timestamps == unique.times[i]
    observed <- as.character(series.id[index])
    ans[i, observed] <- response[index]
  }
  ans <- zoo(ans, unique.times)
  return(ans)
}

ToLong <- function(response, na.rm = TRUE) {
  ## Convert a multiple time series in wide format, to long format.
  ##
  ## Args:
  ##   respponse: A time series matrix (or zoo matrix).  Rows represent time
  ##     points.  Columns are different series.
  ##   na.rm: If TRUE then 
  ##
  ## Returns:
  ##   A data frame in "long" format containing three values:
  ##   - The first column contains the timestamps.
  ##   - The second column contains a factor indicating which column is being
  ##      measured.
  ##   - The third column contains the value of the time series.
  ##
  ## Note:
  ##   This could be done with 'reshape'.  I have reworked things by hand in the
  ##   interest of readability.
  stopifnot(is.matrix(response))
  if (nrow(response) == 0) {
    return(NULL)
  }
  nseries <- ncol(response)
  
  if (is.zoo(response)) {
    timestamps <- index(response)
  } else {
    timestamps <- 1:nrow(response);
  }
  vnames <- colnames(response)
  if (is.null(vnames)) {
    vnames <- base::make.names(1:nseries)
  }

  values <- as.numeric(t(response))
  labels <- rep(vnames, times = nseries)
  timestamps <- rep(timestamps, each = nseries)
  
  ans <- data.frame("time" = timestamps, "series" = labels, "values" = values)
  if (na.rm) {
    missing <- is.na(values)
    ans <- ans[!missing, ]
  }
  return(ans)
}

