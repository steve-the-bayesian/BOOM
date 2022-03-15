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

### Functions for obtaining diagnostics (mainly different flavors of
### residuals) from bsts objects.
### ----------------------------------------------------------------------
residuals.bsts <- function(object,
                           burn = SuggestBurn(.1, object),
                           mean.only = FALSE,
                           ...) {
  ## Args:
  ##   object:  An object of class 'bsts'.
  ##   burn:  The number of iterations to discard as burn-in.
  ##   mean.only: Logical.  If TRUE then the mean residual for each
  ##     time period is returned.  If FALSE then the full posterior
  ##     distribution is returned.
  ##   ...: Not used.  This argument is here to comply with the
  ##     generic 'residuals' function.
  ##
  ## Returns:
  ##   If mean.only is TRUE then this function returns a vector of
  ##   residuals with the same "time stamp" as the original series.
  ##   If mean.only is FALSE then the posterior distribution of the
  ##   residuals is returned instead, as a matrix of draws.  Each row
  ##   of the matrix is an MCMC draw, and each column is a time point.
  ##   The colnames of the returned matrix will be the timestamps of
  ##   the original series, as text.
  if (object$family %in% c("logit", "poisson")) {
    stop("Residuals are not supported for Poisson or logit models.")
  }
  state <- object$state.contributions
  if (burn > 0) {
    state <- state[-(1:burn), , , drop = FALSE]
  }
  state <- rowSums(aperm(state, c(1, 3, 2)), dims = 2)
  if (!object$timestamp.info$timestamps.are.trivial) {
    state <- state[, object$timestamp.info$timestamp.mapping, drop = FALSE]
  }
  residuals <- t(t(state) - as.numeric(object$original.series))
  if (mean.only) {
    residuals <- zoo(colMeans(residuals), index(object$original.series))
  } else {
    residuals <- t(zoo(t(residuals), index(object$original.series)))
  }
  return(residuals)
}
###----------------------------------------------------------------------
bsts.prediction.errors <- function(bsts.object,
                                   cutpoints = NULL,
                                   burn = SuggestBurn(.1, bsts.object),
                                   standardize = FALSE) {
  ## Returns the posterior distribution of the one-step-ahead prediction errors
  ## from the bsts.object.  The errors are computing using the Kalman filter,
  ## and are of two types.
  ##
  ## Purely in-sample errors are computed as a by-product of the Kalman filter
  ## as a result of fitting the model.  These are stored in the bsts.object
  ## assuming the save.prediction.errors argument is TRUE, which is the default.
  ## The in-sample errors are 'in-sample' in the sense that the parameter values
  ## used to run the Kalman filter are drawn from their posterior distribution
  ## given complete data.  Conditional on the parameters in that MCMC iteration,
  ## each 'error' is the difference between the observed y[t] and its
  ## expectation given data to t-1.
  ##
  ## Purely out-of-sample errors can be computed by specifying the 'cutpoints'
  ## argument.  If cutpoints are supplied then a separate MCMC is run using just
  ## data up to the cutpoint.  The Kalman filter is then run on the remaining
  ## data, again finding the difference between y[t] and its expectation given
  ## data to t-1, but conditional on parameters estimated using data up to the
  ## cutpoint.
  ##
  ## Args:
  ##   bsts.object:  An object created by a call to 'bsts'.
  ##   cutpoints: An increasing sequence of integers between 1 and the number of
  ##     time points in the training data for 'bsts.object', or NULL.  If NULL
  ##     then the in-sample one-step prediction errors from the bsts object will
  ##     be extracted and returned.  Otherwise the model will be re-fit with a
  ##     separate MCMC run for each entry in 'cutpoints'.  Data up to each
  ##     cutpoint will be included in the fit, and one-step prediction errors
  ##     for data after the cutpoint will be computed.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   standardize: Logical.  If TRUE then the prediction errors are divided by
  ##     the square root of the one-step-ahead forecast variance.  If FALSE the
  ##     raw errors are returned.
  ##
  ## Returns:
  ##   A list, with entries giving the distribution of one-step prediction
  ##   errors corresponding to individual cutpoints.  Each list entry is a
  ##   matrix, with rows corresponding to MCMC draws, and columns corresponding
  ##   to time points in the data for bsts.object.  If the in-sample prediction
  ##   errors were stored in the original model fit, they will be present in
  ##   list.
  stopifnot(inherits(bsts.object, "bsts"))
  stopifnot(is.numeric(burn), length(burn) == 1, burn < bsts.object$niter)
  stopifnot(is.logical(standardize), length(standardize) == 1)
  if (bsts.object$family %in% c("logit", "poisson")) {
    stop("Prediction errors are not supported for Poisson or logit models.")
  }
  if (HasDuplicateTimestamps(bsts.object)) {
    stop("Prediction errors are not supported for duplicate timestamps.")
  }

  if (standardize && is.null(cutpoints)) {
    ## If standardized errors are desired then the kalman filter will have to be
    ## run, so we need to force the data through the C++ code path.  To do this
    ## we add a cutpoint at the final observation.  If timestamps are trivial
    ## then this is just the length of the original series.  Otherwise it comes
    ## from the mapping in timestamp.info.
    if (bsts.object$timestamp.info$timestamps.are.trivial) {
      cutpoints <- length(bsts.object$original.series)
    } else {
      cutpoints <- max(bsts.object$timestamp.info$timestamp.mapping)
    }
  }

  if (!is.null(cutpoints) && length(cutpoints) > 0) {
    stopifnot(length(cutpoints) <= bsts.object$number.of.time.points,
              all(cutpoints > 0),
              all(cutpoints < bsts.object$number.of.time.points))
    errors <- .Call("analysis_common_r_bsts_one_step_prediction_errors_",
                    bsts.object,
                    as.integer(cutpoints),
                    as.logical(standardize),
                    PACKAGE = "bsts")
  } else {
    errors <- NULL
  }

  if (!is.null(bsts.object$one.step.prediction.errors)) {
    errors$in.sample <- bsts.object$one.step.prediction.errors
  }

  if (burn > 0) {
    for (i in seq_along(errors)) {
      errors[[i]] <- errors[[i]][-(1:burn), , drop = FALSE]
    }
  }

  if (!bsts.object$timestamp.info$timestamps.are.trivial) {
    ## If timestamps are not trivial then there are either duplicate timestamps
    ## or some missing values in the data.  Duplicates were checked above, so
    ## there must be missing observations, which will not be found in
    ## bsts.object$original.series.  The following loop eliminates them from the
    ## set of one step prediction errors.
    mapping <- bsts.object$timestamp.info$timestamp.mapping
    for (i in seq_along(errors)) {
      errors[[i]] <- errors[[i]][, mapping, drop = FALSE]
    }
  }

  attributes(errors)$cutpoints <- cutpoints
  attributes(errors)$timestamps <- bsts.object$timestamp.info$timestamps
  class(errors) <- "bsts.prediction.errors"
  return(errors)
}
