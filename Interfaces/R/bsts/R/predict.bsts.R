# Copyright 2018 Google LLC. All Rights Reserved.
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

predict.bsts <- function(object,
                         horizon = 1,
                         newdata = NULL,
                         timestamps = NULL,
                         burn = SuggestBurn(.1, object),
                         na.action = na.exclude,
                         olddata = NULL,
                         olddata.timestamps = NULL,
                         trials.or.exposure = 1,
                         quantiles = c(.025, .975),
                         seed = NULL,
                         ...) {
  ## Args:
  ##   object:  an object of class 'bsts' created using the function 'bsts'
  ##   newdata: a vector, matrix, or data frame containing the predictor
  ##     variables to use in making the prediction.  This is only required if
  ##     'object' contains a regression component.  If a data frame, it must
  ##     include variables with the same names as the data used to fit 'object'.
  ##     The first observation in newdata is assumed to be one time unit after
  ##     the end of the last data used in fitting 'object', and the subsequent
  ##     observations are sequential time points.  If the regression part of
  ##     'object' contains only a single predictor then newdata can be a vector.
  ##     If 'newdata' is passed as a matrix it is the caller's responsibility to
  ##     ensure that it contains the correct number of columns and that the
  ##     columns correspond to those in object$coefficients.
  ##   timestamps: A vector of time stamps (of the same type as the timestamps
  ##     used to fit 'object'), with one per row of 'newdata' (or element of
  ##     'newdata', if 'newdata' is a vector).  The time stamps give the time
  ##     points as which each prediction is desired.  They must be interpretable
  ##     as integer (0 or larger) time steps following the last time stamp in
  ##     'object'.  If NULL, then the requested predictions are interpreted as
  ##     being at 1, 2, 3, ... steps following the training data.
  ##   horizon: An integer specifying the number of periods into the future you
  ##     wish to predict.  If 'object' contains a regression component then the
  ##     forecast horizon is nrow(X) and this argument is not used.
  ##   burn: An integer describing the number of MCMC iterations in 'object' to
  ##     be discarded as burn-in.  If burn <= 0 then no burn-in period will be
  ##     discarded.
  ##   na.action: A function determining what should be done with missing values
  ##     in newdata.
  ##   olddata: An optional data frame including variables with the same names
  ##     as the data used to fit 'object'.  If 'olddata' is missing then it is
  ##     assumed that the first entry in 'newdata' immediately follows the last
  ##     entry in the training data for 'object'.  If 'olddata' is supplied then
  ##     it will be filtered to get the distribution of the next state before a
  ##     prediction is made, and it is assumed that the first entry in 'newdata'
  ##     comes immediately after the last entry in 'olddata'.
  ##   olddata.timestamps: A set of timestamps corresponding to the observations
  ##     supplied in olddata.  If olddata is not supplied this is not used.  If
  ##     olddata is supplied and this is NULL then trivial timestamps (1, 2,
  ##     ...) will be assumed.  Otherwise this argument behaves like the
  ##     'timestamps' argument to the 'bsts' function.
  ##   trials.or.exposure: For logit or Poisson models, the number of binomial
  ##     trials (or the exposure time) to assume at each time point in the
  ##     forecast period.  This can either be a scalar (if the number of trials
  ##     is to be the same for each time period), or it can be a vector with
  ##     length equal to 'horizon' (if the model contains no regression term) or
  ##     'nrow(newdata)' if the model contains a regression term.
  ##   quantiles: A numeric vector of length 2 giving the lower and upper
  ##     quantiles to use for the forecast interval estimate.
  ##   seed: An integer to use as the C++ random seed.  If NULL then the C++
  ##     seed will be set using the clock.
  ##   ...: Not used.  Present to match the signature of the default predict
  ##     method.
  ##
  ## Returns:
  ##   An object of class 'bsts.prediction', which is a list with the
  ##   following elements:
  ##   mean: A numeric vector giving the posterior mean of the
  ##     predictive distribution at each time point.
  ##   interval: A two-column matrix giving the lower and upper limits
  ##     of the 95% prediction interval at each time point.
  ##   distribution: A matrix of draws from the posterior predictive
  ##     distribution.  Each column corresponds to a time point.  Each
  ##     row is an MCMC draw.
  ##   original.series: The original series used to fit 'object'.
  ##     This is used by the plot method to plot the original series
  ##     and the prediction together.
  stopifnot(inherits(object, "bsts"))
  prediction.data <- .FormatBstsPredictionData(
    object, newdata, horizon, trials.or.exposure, na.action)
  prediction.data$timestamps <- .ExtractPredictionTimestamps(
    object, newdata, timestamps)
  stopifnot(is.numeric(burn), length(burn) == 1, burn < object$niter)
  if (!is.null(olddata)) {
    olddata <- .FormatObservedDataForPredictions(object, olddata, na.action,
      olddata.timestamps)
    original.series <- olddata$response
  } else {
    original.series <- object$original.series
  }

  stopifnot(is.null(seed) || length(seed) == 1)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  predictive.distribution <- .Call("analysis_common_r_predict_bsts_model_",
                                   object,
                                   prediction.data,
                                   burn,
                                   olddata,
                                   seed = seed,
                                   PACKAGE = "bsts")

  ans <- list("mean" = colMeans(predictive.distribution),
              "median" = apply(predictive.distribution, 2, median),
              "interval" = apply(predictive.distribution, 2,
                                 quantile, quantiles),
              "distribution" = predictive.distribution,
              "original.series" = original.series)
  class(ans) <- "bsts.prediction"
  return(ans)
}

###----------------------------------------------------------------------
plot.bsts.prediction <- function(x,
                                 y = NULL,
                                 burn = 0,
                                 plot.original = TRUE,
                                 median.color = "blue",
                                 median.type = 1,
                                 median.width = 3,
                                 interval.quantiles = c(.025, .975),
                                 interval.color = "green",
                                 interval.type = 2,
                                 interval.width = 2,
                                 style = c("dynamic", "boxplot"),
                                 ylim = NULL,
                                 ...) {
  ## Plots the posterior predictive distribution found in the
  ## 'prediction' object.
  ## Args:
  ##   x: An object with class 'bsts.prediction', generated
  ##     using the 'predict' method for a 'bsts' object.
  ##   y: A dummy argument needed to match the signature of the plot()
  ##     generic function.  It is not used.
  ##   burn: The number of observations you wish to discard as burn-in
  ##     from the posterior predictive distribution.  This is in
  ##     addition to the burn-in discarded using predict.bsts.
  ##   plot.original: Logical or numeric.  If TRUE then the prediction
  ##     is plotted after a time series plot of the original series.
  ##     If FALSE, the prediction fills the entire plot.  If numeric,
  ##     then it specifies the number of trailing observations of the
  ##     original time series to plot.
  ##   median.color: The color to use for the posterior median of the
  ##     prediction.
  ##   median.type: The type of line (lty) to use for the posterior median
  ##     of the prediction.
  ##   median.width: The width of line (lwd) to use for the posterior median
  ##     of the prediction.
  ##   interval.quantiles: The lower and upper limits of the credible
  ##     interval to be plotted.
  ##   interval.color: The color to use for the upper and lower limits
  ##     of the 95% credible interval for the prediction.
  ##   interval.type: The type of line (lty) to use for the upper and
  ##     lower limits of the 95% credible inerval for of the
  ##     prediction.
  ##   interval.width: The width of line (lwd) to use for the upper and
  ##     lower limits of the 95% credible inerval for of the
  ##     prediction.
  ##   style: What type of plot should be produced?  A
  ##     DynamicDistribution plot, or a time series boxplot.
  ##   ylim:  Limits on the vertical axis.
  ##   ...: Extra arguments to be passed to PlotDynamicDistribution()
  ##     and lines().
  ## Returns:
  ##   This function is called for its side effect, which is to
  ##   produce a plot on the current graphics device.
  prediction <- x
  if (burn > 0) {
    prediction$distribution <-
        prediction$distribution[-(1:burn), , drop = FALSE]
    prediction$median <- apply(prediction$distribution, 2, median)
    prediction$interval <- apply(prediction$distribution, 2,
                                 quantile, c(.025, .975))
  }
  prediction$interval <- apply(prediction$distribution, 2,
                               quantile, interval.quantiles)

  original.series <- prediction$original.series
  if (is.numeric(plot.original)) {
    original.series <- tail(original.series, plot.original)
    plot.original <- TRUE
  }
  n1 <- ncol(prediction$distribution)

  time <- index(original.series)
  deltat <- tail(diff(tail(time, 2)), 1)

  if (is.null(ylim)) {
    ylim <- range(prediction$distribution, original.series, na.rm = TRUE)
  }

  if (plot.original) {
    pred.time <- tail(time, 1) + (1:n1) * deltat
    plot(time,
         original.series,
         type = "l",
         xlim = range(time, pred.time, na.rm = TRUE),
         ylim = ylim,
         ...)
  } else {
    pred.time <- tail(time, 1) + (1:n1) * deltat
  }

  style <- match.arg(style)
  if (style == "dynamic") {
    PlotDynamicDistribution(curves = prediction$distribution,
                            timestamps = pred.time,
                            add = plot.original,
                            ylim = ylim,
                            ...)
  } else {
    TimeSeriesBoxplot(prediction$distribution,
                      time = pred.time,
                      add = plot.original,
                      ylim = ylim,
                      ...)
  }
  lines(pred.time, prediction$median, col = median.color,
        lty = median.type, lwd = median.width, ...)
  for (i in 1:nrow(prediction$interval)) {
    lines(pred.time, prediction$interval[i, ], col = interval.color,
          lty = interval.type, lwd = interval.width, ...)
  }
  return(invisible(NULL))
}

###---------------------------------------------------------------------------
# Private section
###---------------------------------------------------------------------------

.FormatBstsPredictionData <- function(
    object,
    newdata,
    horizon,
    trials.or.exposure,
    na.action) {
  ## Package the data on which to base the prediction in the form expected by
  ## underlying C++ code.
  ##
  ## Args:
  ##   object:  A bsts model object.
  ##   newdata: The data needed to make future predictions.  In simple Gaussian
  ##     models with no predictors this is not used.  In models with a
  ##     regression component it must be one of the following.
  ##     * A data.frame containing variables with names and types matching those
  ##       used in fitting the original model.
  ##     * A matrix with the number of columns matching object$coefficients.  If
  ##       the number of columns is one too few, an intercept term will be
  ##       added.
  ##     * If object$coefficients is based on a single predictor, a vector can
  ##       be passed.
  ##     newdata can also contain information about binomial trials, poisson
  ##     exposures, or predictors needed for dynamic regression state
  ##     components.
  ##   horizon: An integer giving the number of forecast periods.
  ##   trials.or.exposure: If the model family is poisson or logit, this
  ##     argument specifies the number of binomial trials or the Poisson
  ##     exposure times.  If used, it must be one of the following:
  ##     * A string naming a column in newdata containing the trials or exposure
  ##       field.
  ##     * A single number giving the number of trials or length of exposure
  ##       time to use for all predictions.
  ##     * A vector of numbers to use as the trials or exposure times.
  ##     If the final option is used, its length must be 'horizon'.
  ##
  ## Returns:
  ##   A list of prediction data, suitable for passing to the .Call
  ##   function used in the predict.bsts method.
  if (object$has.regression) {
    predictors <- .ExtractPredictors(object, newdata, na.action = na.action)
    horizon <- nrow(predictors)
  } else {
    predictors <- matrix(rep(1, horizon), ncol = 1)
  }
  horizon <- as.integer(horizon)

  if (object$family == "gaussian" || object$family == "student") {
    if (object$has.regression) {
      ans <- list("predictors" = predictors, horizon = horizon)
    } else {
      ans <- list("horizon" = horizon)
    }
  } else if (object$family == "logit") {
    ans <- list(
      "predictors" = predictors,
      "horizon" = horizon,
      "trials" = .FormatTrialsOrExposure(trials.or.exposure, newdata, horizon))
  } else if (object$family == "poisson") {
    ans <- list(
      "predictors" = predictors,
      "horizon" = horizon,
      "exposure" = .FormatTrialsOrExposure(
        trials.or.exposure, newdata, horizon))
  } else {
    stop("Unrecognized object family in .FormatBstsPredictionData")
  }

  ## If the model object contains any dynamic regression components, add them
  ## here.
  ans <- .ExtractDynamicRegressionPredictors(ans, object, newdata)

  ## Handle the case where there is no static regression, but there is a dynamic
  ## regression.
  if (!is.null(ans$dynamic.regression.predictors)) {
    if (is.null(ans$predictors)
      || (nrow(ans$dynamic.regression.predictors) != nrow(ans$predictors)
        && all(ans$predictors == 1))) {
      ans$predictors <- matrix(rep( 1, nrow(ans$dynamic.regression.predictors)))
    }
  }
  return(ans)
}

###---------------------------------------------------------------------------
.FormatTrialsOrExposure <- function(arg, newdata, horizon) {
  ## Get the number of binomial trials, or Poisson exposure times, for
  ## forecasting binomial or Poisson data.
  ##
  ## Args:
  ##   arg:  Can be one of 3 things:
  ##     * A string naming a column in newdata containing the trials
  ##       or exposure field.
  ##     * A single number giving the number of trials or length of
  ##       exposure time to use for all predictions.
  ##     * A vector of numbers to use as the trials or exposure times.
  ##   newdata: If 'arg' is a string, then newdata must be a data
  ##     frame containing a column with the corresponding name, filled
  ##     with the vector of trials or exposure times to be used.  If
  ##     'arg' is numeric then 'newdata' is not used.
  ##   horizon: An integer giving the number of forecast periods.  This is only
  ##     needed if newdata is NULL.
  ##
  ## Returns:
  ##   A numeric vector of length 'horizon' containing the trials or
  ##   exposure time to use in each foreacst period.
  if (is.data.frame(newdata) || is.matrix(newdata)) {
    horizon <- nrow(newdata)
  } else if (is.zoo(newdata)) {
    if (!is.null(nrow(newdata))) {
      horizon <- nrow(newdata)
    } else {
      horizon <- length(newdata)
    }
  } else if (is.numeric(newdata)) {
    horizon <- length(newdata)
  }
  if (is.character(arg)) {
   arg <- newdata[, arg]
  }
  if (is.integer(arg)) {
    arg <- as.numeric(arg)
  }
  if (!is.numeric(arg)) {
    stop("trials.or.exposure must either be a numeric vector, or the ",
         "name of a numeric column in newdata")
  }
  if (length(arg) == 1) {
    arg <- rep(arg, horizon)
  }
  if (length(arg) != horizon) {
    stop("Length of trials.or.exposure must either be 1, or else match the ",
         "number of forecast periods.")
  }
  return(arg)
}

###---------------------------------------------------------------------------
.FormatObservedDataForPredictions <- function(bsts.object, olddata, na.action,
                                              olddata.timestamps) {
  ## Ensures correct formatting of an 'olddata' argument supplied to
  ## predict.bsts.
  ##
  ## Args:
  ##   bsts.object:  The model object produced by bsts.
  ##   olddata: The object passed as 'olddata' to predict.bsts.  This is often a
  ##     data frame containing the predictor variables and response, but it
  ##     might be just a numeric vector if the model contained no regression
  ##     component.
  ##   na.action:  What to do about NA's?
  ##   olddata.timestamps: The timestamps corresponding to the olddata argument.
  ##     This can be NULL in which case trivial time stamps are assumed.
  ##
  ## Returns:
  ##   A list containing the values of 'response', 'predictors',
  ##   'response.is.observed', and 'timestamp.info', as expected by the
  ##   underlying C++ code.  Additional fields may be added for other model
  ##   families.
  if (bsts.object$has.regression) {
    predictors <- .ExtractPredictors(bsts.object, olddata,
                                     na.action = na.action)
  } else {
    predictors <- NULL
  }
  response <- .ExtractResponse(bsts.object, olddata, na.action = na.action)

  if (bsts.object$family == "gaussian" ||
      bsts.object$family == "student") {
    observed.data <- list("predictors" = predictors,
                          "response" = response)
  } else if (bsts.object$family == "logit") {
    if (is.matrix(response)) {
      trials <- rowSums(response)
      response <- response[, 1]
    } else {
      response <- response > 0
      trials <- rep(1, length(response))
    }
    observed.data <- list("predictors" = predictors,
                          "response" = response,
                          "trials" = trials)
  } else if (bsts.object$family == "poisson") {
    if (is.matrix(response)) {
      exposure <- response[, 2]
      response <- response[, 1]
    } else {
      exposure <- rep(1, length(response))
    }
    observed.data <- list("predictors" = predictors,
                          "response" = response,
                          "exposure" = exposure)
  } else {
    stop("Unknown model family in .FormatObservedDataForPredictions")
  }
  observed.data$response.is.observed <- !is.na(observed.data$response)
  observed.data$timestamp.info <- TimestampInfo(
      observed.data$response, observed.data$predictors, olddata.timestamps)
  return(observed.data)
}

###---------------------------------------------------------------------------
.ExtractPredictionTimestamps <- function(bsts.object,
                                         newdata,
                                         timestamps) {
  ## Args:
  ##   bsts.object: The bsts object representing the posterior distribution of
  ##     the model given training data.
  ##   newdata:  The covariates passed to the predict method.
  ##   timestamps: Either a vector of timestamps (of the same type used to fit
  ##     bsts.object), or NULL.
  ##
  ## Returns:
  ##   If 'timestamps' is non-NULL then a list containing the following
  ##   elements is returned.
  ##   * timestamps: the time stamps indexing the newdata object
  ##   * regular.timestamps: A sequence of time stamps beginning with either
  ##       timestamps[1], or the first time point after the final time stamp in
  ##       bsts.object, whichever comes first.
  ##   * timestamp.mapping: A vector with length equal to nrow(newdata) (or
  ##       length(newdata), if newdata is a vector), giving time stamp of each
  ##       row (or element) of newdata, measured as the the number of time steps
  ##       after the final time stamp in bsts.object.
  ##
  ## If timestamps is NULL then NULL is returned.
  if (!is.null(timestamps)) {
    last.training.time <- max(bsts.object$timestamp.info$regular.timestamps)
    regular.timestamps <- RegularizeTimestamps(
        c(last.training.time, timestamps))
    timestamp.mapping <- zoo::MATCH(timestamps, regular.timestamps) - 1
    if (timestamps[1] > last.training.time) {
      regular.timestamps <- regular.timestamps[-1]
    }
    return(list(timestamps = timestamps,
                regular.timestamps = regular.timestamps,
                timestamp.mapping = as.integer(timestamp.mapping)))
  } else {
    return(NULL)
  }
}

###---------------------------------------------------------------------------
.ExtractDynamicRegressionPredictors <- function(
    prediction.data, bsts.object, dataframe, na.action) {
  ## Args:
  ##   A list of data required for prediction.
  ##   bsts.object: A model object fit by bsts.  The
  ##     state.specification component of this object may or may not
  ##     have one or more DynamicRegression components.
  ##   dataframe: A data frame containing variables with the same
  ##     names and types used to fit bsts.object.
  ##
  ## Returns:
  ##   prediction.data
  dynamic.regression <- sapply(bsts.object$state.specification,
                               inherits,
                               "DynamicRegression")
  if (sum(dynamic.regression) > 1) {
    stop("The model should not contain more than one",
         "dynamic regression component.")
  }
  if (any(dynamic.regression)) {
    ## dynamic.regression is now a list of dynamic regression state
    ## specification objects.
    dynamic.regression <-
        seq_along(bsts.object$state.specification)[dynamic.regression]
    for (d in dynamic.regression) {
      dr.object <- bsts.object$state.specification[[d]]
      predictors <- .ExtractPredictors(dr.object,
                                       dataframe,
                                       xdim = ncol(dr.object$predictors),
                                       na.action)
      prediction.data$dynamic.regression.predictors <- as.matrix(predictors)
      if (prediction.data$horizon == 1) {
        prediction.data$horizon <-
          nrow(prediction.data$dynamic.regression.predictors)
      }
    }
  }
  return(prediction.data)
}

###---------------------------------------------------------------------------
.ExtractPredictors <- function(
    object,
    newdata,
    xdim = NULL,
    na.action) {
  ## Create the matrix of predictors from newdata, using an object's
  ##   * terms
  ##   * xlevels
  ##   * contrasts
  ##
  ## Args:
  ##   object: Either an object created by a call to 'bsts', or a
  ##     dynamic regression state component.
  ##   newdata: The data needed to make future predictions.  In simple
  ##     Gaussian models with no predictors this argument is not used.
  ##     In models with a regression component it must be one of the
  ##     following.
  ##     * a data.frame containing variables with names and types
  ##       matching those used in fitting the original model
  ##     * a matrix with the number of columns matching
  ##       object$coefficients.  If the number of columns is one too
  ##       few, an intercept term will be added.
  ##     * If object$coefficients is based on a single predictor, a
  ##       vector can be passed.
  ##     newdata can also contain predictors needed for dynamic regression
  ##     state components.
  ##
  ##     If the model is multivariate, then newdata must be structured so that
  ##     its first 'nseries' rows describe the 'nseries' time series in the
  ##     first forecast period.  The next 'nseries' rows describe the second
  ##     forecast period, etc.
  ##
  ##   xdim: The dimension of the set of coefficients that will be
  ##     used for prediction.
  ##
  ## Returns:
  ##   The matrix of predictors defined by newdata and the regression
  ##   model structure.
  if (is.null(xdim)) {
    beta <- object$coefficients
    if (is.null(beta)) {
      stop(paste0("Cannot extract predictors for a model with no ",
        "regression component."))
    }
    if (length(dim(beta)) == 2) {
      xdim <- ncol(beta)
    } else if (length(dim(beta)) == 3) {
      xdim <- dim(beta)[3]
    } else {
      stop("'coefficients' element should have dimension 2 or 3.")
    }
  }

  if (is.null(newdata)) {
    stop("You need to supply 'newdata' when making predictions with ",
         "a bsts object that has a regression component.")
  }
  if (is.zoo(newdata)) {
    newdata <- as.data.frame(newdata)
  }
  if (is.data.frame(newdata)) {
    Terms <- delete.response(terms(object))
    newdata.frame <- model.frame.default(
        Terms,
        data = newdata,
        na.action = na.action,
        xlev = object$xlevels)
    data.classes <- data.classes <- attr(Terms, "dataClasses")
    if (!is.null(data.classes)) {
      .checkMFClasses(data.classes, newdata.frame)
    }
    predictors <- model.matrix(
      Terms, newdata.frame, contrasts.arg = object$contrasts)

    if ((inherits(object, "DynamicRegression"))
        && ("(Intercept)" %in% colnames(predictors))) {
      intercept.position <- grep("(Intercept)", colnames(predictors))
      predictors <- predictors[, -intercept.position, drop = FALSE]
    }

    if (nrow(predictors) != nrow(newdata)) {
      warning("Some entries in newdata have missing values, and  will ",
              "be omitted from the prediction.")
    }
    if (ncol(predictors) != xdim) {
      stop("Wrong number of columns in newdata.  ",
           "(Check that variable names match?)")
    }
  } else {
    ## newdata is not a data.frame, so it must be a vector or a
    ## matrix.  Convert it to a matrix for consistent handling.
    predictors <- as.matrix(newdata)
    if (ncol(predictors) == xdim - 1) {
      predictors <- cbind(1, predictors)
    }
    if (ncol(predictors) != xdim) {
      stop(paste("Wrong number of columns in newdata. ",
        "Consider passing a data frame?"))
    }

    na.rows <- rowSums(is.na(predictors)) > 0
    if (any(na.rows)) {
      warning("Entries in newdata containing missing values will be",
              "omitted from the prediction")
      predictors <- predictors[!na.rows, ]
    }
  }
  return(predictors);
}

###---------------------------------------------------------------------------
.ExtractResponse <- function(object, olddata, na.action) {
  if (object$has.regression) {
    Terms <- terms(object)
    bsts.model.frame <- model.frame(Terms,
                                    olddata,
                                    na.action = na.action,
                                    xlev = object$xlevels)
    if (!is.null(data.classes <- attr(Terms, "dataClasses"))) {
      .checkMFClasses(data.classes, bsts.model.frame)
    }
    response <- model.response(bsts.model.frame, "any")
    if (is.matrix(response)) {
      stopifnot(ncol(response) == 2)
    }
  } else if (!is.null(olddata)) {
    response <- olddata
  } else {
    response <- object$original.series
  }
  return(response)
}
