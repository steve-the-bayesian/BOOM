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

.ExtractPredictors <- function(
    object,
    newdata,
    xdim = NULL,
    na.action) {
  ## Create the matrix of predictors from a newdata, using an object's
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
      stop("Wrong number of columns in newdata")
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
    dynamic.regression <-
        seq_along(bsts.object$state.specification)[dynamic.regression]
    for (d in dynamic.regression) {
      dr.object <- bsts.object$state.specification[[d]]
      predictors <- .ExtractPredictors(dr.object,
                                       dataframe,
                                       xdim = ncol(dr.object$predictors),
                                       na.action)
      prediction.data$dynamic.regression.predictors <- as.matrix(predictors)
    }
  }
  return(prediction.data)
}

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

.FormatBstsPredictionData <- function(
    object,
    newdata,
    horizon,
    trials.or.exposure,
    na.action) {
  ## Args:
  ##   object:  A bsts model object.
  ##   newdata: The data needed to make future predictions.  In simple
  ##     Gaussian models with no predictors this is not used.  In
  ##     models with a regression component it must be one of the
  ##     following.
  ##     * a data.frame containing variables with names and types
  ##       matching those used in fitting the original model
  ##     * a matrix with the number of columns matching
  ##       object$coefficients.  If the number of columns is one too
  ##       few, an intercept term will be added.
  ##     * If object$coefficients is based on a single predictor, a
  ##       vector can be passed.
  ##     newdata can also contain information about binomial trials,
  ##     poisson exposures, or predictors needed for dynamic regression
  ##     state components.
  ##   horizon: An integer giving the number of forecast periods.
  ##   trials.or.exposure: If the model family is poisson or logit,
  ##     this argument specifies the number of binomial trials or the
  ##     Poisson exposure times.  If used, it must be one of the
  ##     following:
  ##     * A string naming a column in newdata containing the trials
  ##       or exposure field.
  ##     * A single number giving the number of trials or length of
  ##       exposure time to use for all predictions.
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

  if (object$family == "gaussian" || object$family == "student") {
    if (object$has.regression) {
      ans <- list("predictors" = predictors)
    } else {
      ans <- list("horizon" = as.integer(horizon))
    }
  } else if (object$family == "logit") {
    ans <- list(
      "predictors" = predictors,
      "trials" = .FormatTrialsOrExposure(trials.or.exposure, newdata, horizon))
  } else if (object$family == "poisson") {
    ans <- list(
      "predictors" = predictors,
      "exposure" = .FormatTrialsOrExposure(
        trials.or.exposure, newdata, horizon))
  } else {
    stop("Unrecognized object family in .FormatBstsPredictionData")
  }
  return(.ExtractDynamicRegressionPredictors(ans, object, newdata))
}

.FormatMultivariatePredictionData <- function(object, newdata, horizon, na.action) {
  if (object$has.regression) {
    predictors <- .ExtractPredictors(
      object, newdata, na.action = na.action)
    horizon <- nrow(predictors) / object$nseries
  } else {
    predictors <- matrix(rep(1, horizon * object$nseries), ncol = 1)
  }

  return(list("predictors" = predictors,
    "horizon" = horizon))
}

.FormatTrialsOrExposure <- function(arg,
                                    newdata,
                                    horizon = nrow(newdata)) {
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
  ##   horizon:  An integer giving the number of forecast periods.
  ##
  ## Returns:
  ##   A numeric vector of length 'horizon' containing the trials or
  ##   exposure time to use in each foreacst period.
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
  observed.data$timestamp.info <- .ComputeTimestampInfo(
      observed.data$response, observed.data$predictors, olddata.timestamps)
  return(observed.data)
}
