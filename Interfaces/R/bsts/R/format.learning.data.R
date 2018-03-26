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

.FormatBstsDataAndOptions <- function(family, response, predictors,
                                      model.options, timestamp.info) {
  ## This function is part of the implementation for bsts.  It puts
  ## the data and model options into the format expected by the
  ## underlying C++ code.
  ##
  ## Args:
  ##   family: String naming the model family for the observation
  ##     equation.
  ##   response: The vector (or in some cases two-column matrix) of
  ##     responses to be modeled.
  ##   predictors: The matrix of predictor variables.  This can be
  ##     NULL if the model has no regression component.
  ##   model.options:  Options returned by BstsOptions().
  ##   timestamp.info:  The object returned by .ComputeTimestampInfo.
  ##
  ## Returns:  A list with two elements:  data.list and model.options.

  if (family != "gaussian" && model.options$bma.method == "ODA") {
    warning("Orthoganal data augmentation is not available with a",
            "non-Gaussian model family.  Switching to SSVS.")
    model.options$bma.method <- "SSVS"
  }

  if (family %in% c("gaussian", "student")) {
    if (is.matrix(response) && ncol(response) != 1) {
      stop("Matrix responses only work for logit and Poisson models.  ",
           "Did you mean to specify a different model family?")
    }
    data.list <- list(response = as.numeric(response),
                      predictors = predictors,
                      response.is.observed = !is.na(response))
  } else if (family == "logit") {
    ## Unpack the vector of trials.  If 'response' is a 2-column
    ## matrix then the first column is the vector of success counts
    ## and the second is the vector of failure counts.  Otherwise y
    ## is just a vector, and the vector of trials should just be a
    ## column of 1's.
    if (!is.null(dim(response)) && length(dim(response)) > 1) {
      ## Multi-dimensional arrays are not allowed, and the matrix must
      ## have 2 columns.
      stopifnot(length(dim(response)) == 2, ncol(response) == 2)
      ## Success counts are in the first column, and failure counts
      ## are in the second, so you get trials by adding them up.
      trials <- response[, 1] + response[, 2]
      response <- response[, 1]
    } else {
      ## If 'response' is a single column then 'trials' is implicitly
      ## a vector of all 1's so 'response' is binary, and there are
      ## multiple ways to encode binary data.  The following line
      ## converts y's which are TRUE/FALSE, 1/0 or 1/-1 into our
      ## preferred 1/0 encoding.
      response <- response > 0
      trials <- rep(1, length(response))
    }
    stopifnot(all(trials > 0, na.rm = TRUE),
              all(response >= 0, na.rm = TRUE),
              all(trials >= response, na.rm = TRUE))
    stopifnot(all(abs(response - as.integer(response)) < 1e-8, na.rm = TRUE))
    stopifnot(all(abs(trials - as.integer(trials)) < 1e-8, na.rm = TRUE))
    data.list <- list(response = as.numeric(response),
                      trials = trials,
                      predictors = predictors,
                      response.is.observed = !is.na(response))
    ## TODO: consider exposing clt.threshold as an option
    model.options$clt.threshold <- as.integer(3)
  } else if (family == "poisson") {
    if (!is.null(dim(response)) && length(dim(response)) > 1) {
      ## Multi-dimensional arrays are not allowed, and the matrix must
      ## have 2 columns.
      stopifnot(length(dim(response)) == 2, ncol(response) == 2)
      ## If the user passed a formula like "cbind(counts, exposure) ~
      ## x", then response will be a two column matrix
      exposure <- response[, 2]
      response <- response[, 1]
    } else {
      exposure <- rep(1, length(response))
    }
    stopifnot(is.numeric(response))
    stopifnot(all(exposure > 0, na.rm = TRUE),
              all(response >= 0, na.rm = TRUE))
    stopifnot(all(abs(response - as.integer(response)) < 1e-8, na.rm = TRUE))
    data.list <- list(response = as.numeric(response),
                      exposure = exposure,
                      predictors = predictors,
                      response.is.observed = !is.na(response))
  } else {
    stop("Unrecognized value for 'family' argument in bsts.")
  }
  data.list$timestamp.info <- timestamp.info
  return(list(data.list = data.list, model.options = model.options))
}
