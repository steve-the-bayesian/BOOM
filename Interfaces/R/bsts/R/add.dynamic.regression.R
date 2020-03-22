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

DynamicRegressionRandomWalkOptions <- function(
    sigma.prior = NULL,
    sdy = NULL,
    sdx = NULL) {
  ## Args:
  ##   sigma.prior: This can either be a single object of class 'SdPrior', a
  ##     list of such priors, or NULL.  A single SdPrior will get expanded to a
  ##     list.  Each list element describes the prior distribution for the
  ##     innovation variance of the corresponding dynamic coefficient.  If NULL
  ##     then a default prior will be assigned.
  ##   sdy: The standard deviation of the response variable.  This is used to
  ##     scale the default prior if sigma.prior is NULL.  It is unused
  ##     otherwise.
  ##   sdx: A vector containing the standard deviations of the predictor
  ##     variables multiplied by the dynamic regression coefficients.  This is
  ##     used to scale the default prior if sigma.prior is NULL.  It is unused
  ##     otherwise.
  ## Returns:
  ##   An object that can be passed to AddDynamicRegression as the model.options argument.
  if (!is.null(sigma.prior)) {
    # sigma.prior must either be an SdPrior or a list of SdPrior objects.
    stopifnot(inherits(sigma.prior, "SdPrior")
      || is.list(sigma.prior) && all(sapply(sigma.prior, inherits, "SdPrior")))
  } else {
    if (is.null(sdy) || is.null(sdx)) {
      stop("If 'sigma.prior' is NULL then both 'sdy' and 'sdx' must be supplied.")
    }
    if (any(sdx <= 0)) {
      stop("All elements of sdx must be positive.")
    }

    number.of.coefficients <- length(sdx)
    sigma.prior <- list()
    for (i in 1:number.of.coefficients) {
      sigma.prior[[i]] <- SdPrior(.01 * sdy / sdx[i], 1)
    }
  }
  ans <- list(sigma.prior = sigma.prior)
  class(ans) <- c("DynamicRegressionRandomWalkOptions",
    "DynamicRegressionOptions")
  return(ans)
}

DynamicRegressionHierarchicalRandomWalkOptions <- function(
    sdy = NULL,
    sigma.mean.prior = NULL,
    shrinkage.parameter.prior = GammaPrior(a = 10, b = 1),
    sigma.max = NULL) {
  ## See below for details.
  ## Args:
  ##   sdy: The standard deviation of the response variable.  This is used to
  ##     scale default priors and set sigma.max if those are not supplied.  If
  ##     all other arguments are specified (and non-NULL) then sdy is not used.
  ##   sigma.mean.prior: An object inheriting from 'DoubleModel' giving the
  ##     prior distribution for the mean of the 'sigma' parameters controlling
  ##     the variation of the random walks.  (This is the prior on sqrt(b/a) in
  ##     the Details section below.  A prior that puts is mass on small numbers
  ##     will encourage regression coefficients that don't change very much.
  ##   shrinkage.parameter.prior: An object inheriting from 'DoubleModel'.  This
  ##     is the prior on 'a' in the Details section below, which can be thought
  ##     of as a "prior sample size" for learning each coefficient's sigma
  ##     parameter.  A prior that puts its mass on small numbers encourages each
  ##     sigma[i] to be learned independently.  A prior that puts all its mass
  ##     on very large numbers effectively forces all the sigma[i]'s to be the
  ##     same.
  ##   sigma.max: The largest supported value of each sigma[i].  Truncating the
  ##     support of sigma can keep ill-conditioned models from crashing.  This
  ##     must be a positive number (Inf is okay), or NULL.  A NULL value will
  ##     set sigma.max = sd(y), which is a substantially larger value than one
  ##     would expect, so in well behaved models this constraint will not affect
  ##     the analysis.
  ##
  ## Returns:
  ##   An object indicating the model and prior assumed for the dynamic
  ##   regression coefficients.
  ##
  ## Details:
  ##   The model assumes each regression coefficient follows a random walk, with
  ##   an innovation variance given by a hierarchical model.
  ##
  ##      beta[i, t] ~ N(beta[i, t-1], sigsq[i] / variance_x[i])
  ##  1.0 / sigsq[i] ~ TruncatedGamma(a, b, min)
  ##
  ##   That is, each coefficient has its own variance term, which is
  ##   scaled by the variance of the i'th column of X.  The parameters
  ##   of the hyperprior are interpretable as follows.
  ##   * sqrt(b/a) is the typical amount that a coefficient might change in a
  ##     single time period,
  ##   * 'a' is the 'sample size' or 'shrinkage parameter' measuring the degree
  ##     of similarity in sigma[i] among the arms.
  ##   * min: is the smallest supported value of 1/sigsq (so that 1/min is the
  ##     largest supported value of sigsq).  In models with sparse or poorly
  ##     supported data this value is needed to keep sigsq from marching off to
  ##     infinity.
  ##
  ##   In most cases we hope b/a is small, so that sigma[i]'s will be small and
  ##   the series will be forecastable.  We also hope that 'a' is large because
  ##   it means that the sigma[i]'s will be similar to one another, increasing
  ##   the amount of shared information.
  ##
  ##   The default prior distribution is a pair of independent Gamma priors for
  ##   sqrt(b/a) and a.  The mean of sigma[i] is set to .01 * sd(y) with shape
  ##   parameter equal to 1.  The mean of the shrinkage parameter is set to 10,
  ##   but with shape parameter equal to 1.
  if (is.null(sigma.mean.prior)) {
    ## Set sigma.mean.prior to a default value.
    sigma.mean.prior <- GammaPrior(prior.mean = 0.01 * sdy^2, a = 1)
  }
  stopifnot(inherits(sigma.mean.prior, "DoubleModel"))
  stopifnot(inherits(shrinkage.parameter.prior, "DoubleModel"))

  if (is.null(sigma.max)) {
    sigma.max <- sdy
  }
  stopifnot(is.numeric(sigma.max),
            length(sigma.max) == 1,
            sigma.max > 0)

  ans <- list(sigma.mean.prior = sigma.mean.prior,
              shrinkage.parameter.prior = shrinkage.parameter.prior,
              sigma.max = sigma.max)
  class(ans) <- c("DynamicRegressionHierarchicalRandomWalkOptions",
                  "DynamicRegressionOptions")
  return(ans)
}

DynamicRegressionArOptions <- function(lags = 1, sigma.prior = SdPrior(1, 1)) {
  ## Args:
  ##   lags: The number of lags in the AR(p) process describing the model
  ##     coefficients.
  ##   sigma.prior: The prior distribution for the innovation variance.  This
  ##     can either be an object of class SdPrior, or a list of such objects.
  ##     If a single SdPrior is passed then the same prior will be used for all
  ##     coefficients.  If a list of SdPrior's is passed then its length must
  ##     match the number of coefficients, and each entry will be the prior for
  ##     the corresponding coefficient.
  ##
  ## Returns:
  ##   When the returned object is passed to AddDynamicRegression it signals
  ##   that the regression coefficients should be modeled using an
  ##   AR(p) process.
  ##
  ## Details:
  ##   The model assumes that dynamic regression coefficients evolve according
  ##   to independent AR(p) processes.
  ##
  ##     beta[i, t] = phi[i, 1] * beta[i, t-1]
  ##                  + phi[i, 2] * beta[i, t-2]
  ##                  + ... + epsilon[i, t] ~ N(0, sigma^2[i] / E(x[i]^2))
  ##
  ##   The prior distribution for the phi[i, ] terms is uniform over the
  ##   stationary region for an AR(p) process.  The prior distribution for
  ##   sigma^2 is given by sigma.prior.
  lags <- as.integer(lags)
  stopifnot(lags > 0)

  ## We don't see the number of coefficients here, so the length of the list
  ## will be checked later in .CheckModelOptions.
  stopifnot(inherits(sigma.prior, "SdPrior") ||
            is.list(sigma.prior) && all(sapply(sigma.prior, inherits, "SdPrior")))
  ans <- list(lags = lags,
              sigma.prior = sigma.prior)
  class(ans) <- c("DynamicRegressionArOptions", "DynamicRegressionOptions")
  return(ans)
}

AddDynamicRegression <- function(
    state.specification,
    formula,
    data,
    model.options = NULL,
    sigma.mean.prior.DEPRECATED = NULL,
    shrinkage.parameter.prior.DEPRECATED = GammaPrior(a = 10, b = 1),
    sigma.max.DEPRECATED = NULL,
    contrasts = NULL,
    na.action = na.pass) {
  ## Add a dynamic regression component to the state specification of a bsts
  ## model.  A dynamic regression is a regression model where the coefficients
  ## change over time according to a random walk.
  ##
  ## Args:
  ##   state.specification: A list with elements created by AddLocalLinearTrend,
  ##     AddSeasonal, and similar functions for adding components of state.
  ##   formula: A formula describing the regression portion of the relationship
  ##     between y and X. If no regressors are desired then the formula can be
  ##     replaced by a numeric vector giving the time series to be modeled.
  ##     Missing values are not allowed.
  ##   data: An optional data frame, list or environment (or object coercible by
  ##     ‘as.data.frame’ to a data frame) containing the variables in the model.
  ##     If not found in ‘data’, the variables are taken from
  ##     ‘environment(formula)’, typically the environment from which ‘bsts’ is
  ##     called.
  ##   model.options: An object inheriting from 'DynamicRegressionOptions'
  ##     giving the specific transition model for the dynamic regression
  ##     coefficients, and the prior distribution for any hyperparameters
  ##     associated with the transition model.
  ##   contrasts: An optional list. See the ‘contrasts.arg’ of
  ##     ‘model.matrix.default’.  This argument is only used if a model formula
  ##     is specified.  It can usually be ignored even then.
  ##   na.action: What to do about missing values.  The default is to
  ##     allow missing responses, but no missing predictors.  Set this
  ##     to na.omit or na.exclude if you want to omit missing
  ##     responses altogether.
  ##
  ## Details:
  if (missing(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))

  ## Check for previous dynamic regression components.
  if (any(sapply(state.specification, inherits, "DynamicRegression"))) {
    ## TODO: Document why there can be only one dynamic regression.
    ## It is conceivable that different subsets of the data might want different
    ## prior distributions, or something like that.
    stop("The model cannot contain more than one dynamic regression component.")
  }

  ## Build the model frame that model.matrix can use to build the
  ## design matrix for the dynamic regression.  This code mimics the
  ## code in 'lm'.
  function.call <- match.call()
  my.model.frame <- match.call(expand.dots = FALSE)
  frame.match <- match(c("formula", "data", "na.action"),
                       names(my.model.frame), 0L)
  my.model.frame <- my.model.frame[c(1L, frame.match)]
  my.model.frame$drop.unused.levels <- TRUE

  # In an ordinary regression model the default action for NA's is to
  # delete them.  This makes sense in ordinary regression models, but
  # is dangerous in time series, because it artificially shortens the
  # time between two data points.  If the user has not specified an
  # na.action function argument then we should use na.pass as a
  # default, so that NA's are passed through to the underlying C++
  # code.
  if (! "na.action" %in% names(my.model.frame)) {
    my.model.frame$na.action <- na.pass
  }
  my.model.frame[[1L]] <- as.name("model.frame")
  my.model.frame <- eval(my.model.frame, parent.frame())
  model.terms <- attr(my.model.frame, "terms")
  predictors <- model.matrix(model.terms, my.model.frame, contrasts)
  if ("(Intercept)" %in% colnames(predictors)) {
    intercept.position <- grep("(Intercept)", colnames(predictors))
    predictors <- predictors[, -intercept.position, drop = FALSE]
  }

  sdx <- function(x) return(sqrt(var(x, na.rm = TRUE)))
  predictor.sd <- apply(predictors, 2, sdx)
  constant.predictors <- predictor.sd <= 0.0
  if (any(constant.predictors)) {
    bad.ones <- colnames(predictors)[constant.predictors]
    msg <- paste("The following predictors had zero standard deviation.",
                 "Expect poor results.\n", paste(bad.ones, collapse = "\n"))
    warning(msg)
  }

  if (any(is.na(predictors))) {
    stop("NA's are not allowed in the predictor matrix.")
  }

  stopifnot(ncol(predictors) >= 1)
  stopifnot(nrow(predictors) >= 1)

  # TODO:  Do you want to ensure that x and y conform?
  # TODO:  handle missing data

  model.options <- .CheckModelOptions(
      model.options = model.options,
      predictor.sd = predictor.sd,
      sigma.max.DEPRECATED = sigma.max.DEPRECATED,
      sigma.mean.prior.DEPRECATED = sigma.mean.prior.DEPRECATED,
      shrinkage.parameter.prior.DEPRECATED = shrinkage.parameter.prior.DEPRECATED,
      my.model.frame = my.model.frame)

  state.component <- list(name = "dynamic",
                          predictors = predictors,
                          size = ncol(predictors),
                          terms = model.terms,
                          model.options = model.options,
                          xlevels = .getXlevels(model.terms, my.model.frame),
                          contrasts = attr(predictors, "contrasts"))
  class(state.component) <- c("DynamicRegression", "StateModel")

  state.specification[[length(state.specification) + 1]] <- state.component
  return(state.specification)
}

.CheckModelOptions <- function(model.options,
                               predictor.sd,
                               sigma.max.DEPRECATED,
                               sigma.mean.prior.DEPRECATED,
                               shrinkage.parameter.prior.DEPRECATED,
                               my.model.frame) {
  if (is.null(model.options)) {
    if (!is.null(sigma.max.DEPRECATED)
        || !is.null(sigma.mean.prior.DEPRECATED)) {
      warning("Please specify the prior for a dynamic regression through the ",
              "'model.options' argument.  Arguments marked DEPRECATED will ",
              "be removed in a future version of bsts.")
    }
    response <- model.response(my.model.frame)
    if (is.matrix(response) && ncol(response) > 1) {
      stop("Models with matrix-valued responses need to specify ",
           "'model.options' directly.")
    }
    sdy <- sqrt(var(response, na.rm = TRUE))
    model.options <- DynamicRegressionRandomWalkOptions(
      sdy = sdy,
      sdx = predictor.sd,
      NULL)
  }
  stopifnot(inherits(model.options, "DynamicRegressionOptions"))

  if (inherits(model.options, "DynamicRegressionArOptions")) {
    if (inherits(model.options$sigma.prior, "SdPrior")) {
      model.options$sigma.prior <- RepList(
        model.options$sigma.prior, length(predictor.sd))
    }
    stopifnot(is.list(model.options$sigma.prior),
              all(sapply(model.options$sigma.prior, inherits, "SdPrior")),
              length(model.options$sigma.prior) == length(predictor.sd))
  }
  return(model.options)
}
