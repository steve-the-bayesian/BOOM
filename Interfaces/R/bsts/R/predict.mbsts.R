# Copyright 2019 Steven L. Scott. All Rights Reserved.
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

predict.mbsts <- function(object,
                          horizon = 1,
                          newdata = NULL,
                          timestamps = NULL,
                          burn = SuggestBurn(.1, object),
                          na.action = na.exclude,
                          quantiles = c(.025, .975),
                          seed = NULL,
                          ...) {
  ## Args:
  ##   object:  an object of class 'mbsts' created using the function 'mbsts'
  ##   horizon: An integer specifying the number of periods into the future you
  ##     wish to predict.  If 'object' contains a regression component then the
  ##     forecast horizon is nrow(X) and this argument is not used.
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
  ##   burn: An integer describing the number of MCMC iterations in 'object' to
  ##     be discarded as burn-in.  If burn <= 0 then no burn-in period will be
  ##     discarded.
  ##   na.action: A function determining what should be done with missing values
  ##     in newdata.
  ##   quantiles: A numeric vector of length 2 giving the lower and upper
  ##     quantiles to use for the forecast interval estimate.
  ##   seed: An integer to use as the C++ random seed.  If NULL then the C++
  ##     seed will be set using the clock.
  ##   ...: Not used.  Present to match the signature of the default predict
  ##     method.
  ##
  ## Returns:
  ##   An object of class 'mbsts.prediction', which is a list containing the
  ##   following elements:
  ##   - mean: A numeric matrix giving the posterior mean of the predictive
  ##       distribution at each time point.
  ##   - interval: An nseries x time x 2 array giving the lower and upper limits
  ##       of the 95% prediction interval at each time point.
  ##   - distribution: A [niter x nseries x time] array of draws from the
  ##       posterior predictive distribution.
  ##   - original.series: The original series used to fit 'object', in wide
  ##       format.  This is used by the plot method to plot the original series
  ##       and the prediction together.
  stopifnot(inherits(object, "mbsts"))
  prediction.data <- .FormatMultivariatePredictionData(
    object, newdata, horizon, na.action, timestamps)
  stopifnot(is.numeric(burn), length(burn) == 1, burn < object$niter)

  stopifnot(is.null(seed) || length(seed) == 1)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }
  predictive.distribution <- .Call(
    "analysis_common_r_predict_multivariate_bsts_model_",
    object,
    prediction.data,
    burn,
    seed = seed,
    PACKAGE = "bsts")

  ans <- list(
    "mean" = apply(predictive.distribution, c(2, 3), mean),
    "median" = apply(predictive.distribution, c(2, 3), median),
    "interval" = aperm(
      apply(predictive.distribution, c(2, 3), quantile, quantiles),
      c(2, 1, 3)),
    "distribution" = predictive.distribution,
    "original.series" = LongToWide(
      object$original.series,
      series.id = object$series.id,
      timestamps = object$timestamp.info$timestamps))
  class(ans) <- "mbsts.prediction"
  series.names <- colnames(ans$original.series)
  rownames(ans$mean) <- series.names
  rownames(ans$median) <- series.names
  dimnames(ans$interval) <- list(series.names, NULL, NULL)
  dimnames(ans$distribution) <- list(NULL, series.names, NULL)
  return(ans)
}

###---------------------------------------------------------------------------
# Private section.  Also see related code in predict.bsts.R.
###---------------------------------------------------------------------------

.FormatMultivariatePredictionData <- function(
    object, newdata, horizon, na.action, timestamps) {
  ## Args:
  ##   object:  An mbsts model object on which to base the prediction.
  ##   newdata: A matrix or data frame containing the variables needed to make
  ##     predictions at future values.  If a data frame, it must contain the
  ##     variables of the same names and types as those appearing in the
  ##     'formula' argument used to creat 'object.'  If a matrix, the number of
  ##     columns must match object$coefficients.  If the number of columns is
  ##     one too few then an intercept term will be added in the first column.
  ##     The data must be organized so that the first 'object$nseries' rows correspond
  ##     to the first forecast time, the next 'nseries' rows to the second
  ##     forecast time, etc.
  ##   horizon: This argument is only used if 'object' contains no regression
  ##     component.  An integer giving the desired number of forecast periods.
  ##   na.action: Passed to .ExtractPredictors (where it is passed to
  ##     model.matrix).
  ##
  ## Returns:
  ##   A list containing the prediction data, in a format suitable for passing
  ##   to the .Call function call in predict.mbsts.
  if (object$has.regression) {
    predictors <- .ExtractPredictors(
      object, newdata, na.action = na.action)
    horizon <- nrow(predictors) / object$nseries
  } else {
    predictors <- matrix(rep(1, horizon * object$nseries), ncol = 1)
  }
  timestamps <- .ExtractPredictionTimestamps(object, newdata, timestamps)
  ans <- list("predictors" = predictors,
    "horizon" = horizon,
    "timestamps" = timestamps)

  ## TODO(when dynamic regression is added to mbsts, add a call similar to
  ## .ExtractDynamicRegressionPredictors.  See the example in predict.bsts.R.
  return(ans)
}
