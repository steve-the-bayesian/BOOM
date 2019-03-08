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

mbsts <- function(formula,
                  state.specification,
                  data,
                  prior = NULL,
                  contrasts = NULL,
                  na.action = na.pass,
                  niter,
                  ping = niter / 10,
                  seed = NULL,
                  ...) {
  check.nonnegative.scalar(niter)
  check.scalar.integer(ping)
  stopifnot(is.null(seed) || length(seed) == 1)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }
  has.regression <- !is.numeric(formula)
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

    # The response must be a matrix of class zoo, with rows corresponding to
    # time stamps, and columns corresponding to observed data.  Many elements
    # of the response matrix may be missing, especially if the time series are
    # not aligned, or did not all start from the same date.
    stopifnot(is.matrix(response))
    stopifnot(nrow(response) == nrow(predictors))
  } else {
    response <- formula
    predictors <- NULL
  }
  stopifnot(is.zoo(response))

  if (missing(data)) {
    # This should be handled in the argument list by setting a default argument
    # data = NULL, but doing that messes up the "regression black magic" section
    # above.
    data <- NULL
  }

  timestamp.info <- .ComputeTimestampInfo(response, data, NULL)
  data.list <- list(response = response,
    predictors = predictors,
    response.is.observed = is.na(response))
    
  ###########################################################################
  # Can't do this part until we know what priors look like.
  ###########################################################################
  
  if (is.null(prior)) {
    stop("Need to implement a default prior")
  }

  if (has.regression) {
    ##########  
  }

  ans <- .Call("analysis_common_r_fit_multivariate_bsts_model_",
    data.list,
    state.specification,
    prior,
    NULL,  # slot for model.options.
    niter,
    ping,
    seed)

  ### Next do cleanup.
  
}
