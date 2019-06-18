# Copyright 2011 Google LLC. All Rights Reserved.
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

AddSharedLocalLevel <- function(state.specification,
                                response,
                                nfactors, 
                                coefficient.prior = NULL,
                                initial.state.prior = NULL,
                                timestamps = NULL,
                                series.id = NULL,
                                sdy,
                                ...) {
  ## A local level model for multivariate time series.  If the state of the
  ## model is alpha[t] then the state equation is
  ##
  ##            y[t] = Z * alpha[t] + epsilon[t]
  ##    alpha[t + 1] = alpha[t] + eta[t]
  ##
  ## For identification purposes, the variance of eta[t] is the identity matrix,
  ## and the coefficient matrix Z It is zero above the diagonal.  This means
  ## that the first time series is only affected by the first factor.  The
  ## second is affected by the first and second factors, etc.  
  ##
  ## Args:
  ##   state.specification: A list of state components to which a shared local
  ##     level model will be added.  If omitted, an empty list is assumed.
  ##   y: A numeric matrix containing the time series to be modeled.  Rows are
  ##     time points and columns are variables.  This argument can be omitted if
  ##     sdy is provided.
  ##   nfactors: An integer giving the number of latent factors in the model.
  ##   coefficient.prior: An object (or a list of objects) inheriting from
  ##     SpikeSlabPriorBase.  If a list is passed it must have 'nseries'
  ##     elements, where 'nseries' is the number of time series being modeled.
  ##     If a single object is passed it will be copied into a list of 'nseries'
  ##     identical prior objects.  List element i specifies the prior
  ##     distribution on the set of observation coefficients for time series i.
  ##     Note that identifiability constriants will be imposed by underlying
  ##     code, so that if series k < 'nfactors', only the first 'k' factors will
  ##     have positive prior probability for series k.
  ##   initial.state.prior: An object created by MvnPrior giving the prior
  ##     distribution on the values of the initial state (i.e. the state as of
  ##     the first observation).
  ##   timestamps: A vector of timestamps indicating the time of each observation.
  ##     If "wide" data are used this argument is optional, as the timestamps will
  ##     be inferred from the rows of the data matrix.
  ##   series.id: A factor-like object indicating the time series to which each
  ##     observation belongs.  This is only necessary for data in "long" format.
  ##   sdy: A vector giving the sample standard deviation for each column in y.
  ##     This will be ignored if y is provided, or if both sigma.prior and
  ##     initial.state.prior are supplied directly.
  ##   ...: Extra arguments passed to ConditionalZellnerPrior, as a prior on the
  ##     observation coefficients.
  ##
  ## Returns:
  ##   state.specification, after appending the information necessary
  ##   to define a shared local level model.
  if (missing(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))
  stopifnot(is.numeric(nfactors), length(nfactors) == 1, nfactors >= 1)
  
  if (!missing(response)) {
    stopifnot(is.numeric(response))
    if (is.matrix(response)) {
      response.matrix <- response
    } else {
      response.matrix <- LongToWide(response, series.id, timestamps)
    }
    stopifnot(is.matrix(response.matrix))
    sdy <- apply(response.matrix, 2, sd, na.rm = TRUE)
    series.names <- colnames(response.matrix)
  }
  stopifnot(is.numeric(sdy), all(sdy > 0))
  nseries <- length(sdy)
  if (nseries < 1) {
    stop("There are no time series to model.")
  }

  nfactors <- as.integer(nfactors)
  stopifnot(length(nfactors) == 1)
  
  ##----------------------------------------------------------------------
  # Set the prior on the observation coefficients.
  ##----------------------------------------------------------------------
  # The coefficients Z satisfy Y[t] = Z * alpha, so the coefficients have
  # 'nseries' rows and 'nfactors' columns.
  if (is.null(coefficient.prior)) {
    coefficient.prior <- list()
    for (i in 1:nseries) {
      ## Normally regression coefficients are centered around zero.  In this
      ## case it makes sense to center around 1.  Really this should be a
      ## hierarchical prior where it makes sense to center the hyperprior around
      ## 1.  TODO(steve)
      coefficient.prior[[i]] <- ConditionalZellnerPrior(
        nfactors,
        optional.coefficient.estimate = rep(1, nfactors),
        ...)
    }
  }
  if (inherits(coefficient.prior, "ConditionalZellnerPrior")) {
    coefficient.prior <- RepList(coefficient.prior, nseries)
  }
  stopifnot(is.list(coefficient.prior),
    all(sapply(coefficient.prior, inherits, "ConditionalZellnerPrior")))
  
  ##----------------------------------------------------------------------
  ## Set the prior on the initial state.
  ##----------------------------------------------------------------------
  if (missing(initial.state.prior)) {
    # When specifying the Matrix in MvnPrior, nrow and ncol must be specified to
    # thwart diag behaving differently when passed a scalar.
    initial.state.prior <- MvnPrior(
      mean = rep(0, nfactors),
      variance = diag(rep(max(sdy), nfactors),
        nrow = nfactors,
        ncol = nfactors))
  }
  stopifnot(inherits(initial.state.prior, "MvnPrior"),
    length(initial.state.prior$mean) == nfactors)

  level <- list(name = "trend",
    coefficient.priors = coefficient.prior,
    initial.state.prior = initial.state.prior,
    series.names = colnames(response.matrix),
    size = nfactors)
  class(level) <- c("SharedLocalLevel", "SharedStateModel")

  state.specification[[length(state.specification) + 1]] <- level
  return(state.specification)
}

#==============================================================================
AddLocalLevel <- function(state.specification,
                          y,
                          sigma.prior,
                          initial.state.prior,
                          sdy,
                          initial.y) {
  ## Adds a local level model (see Harvey, 1989, or Durbin and Koopman
  ## 2001) to a state space model specification.
  ## Args:
  ##   state.specification: A list of state components.  If omitted,
  ##     an empty list is assumed.
  ##   y:  A numeric vector.  The time series to be modeled.
  ##   sigma.prior: An object created by SdPrior.  This is the prior
  ##     distribution on the standard deviation of the level
  ##     increments.
  ##   initial.state.prior: An object created by NormalPrior.  The
  ##     prior distribution on the values of the initial state
  ##     (i.e. the state of the first observation).
  ##   sdy: The standard deviation of y.  This will be ignored if y is
  ##     provided, or if both sigma.prior and initial.state.prior are
  ##     supplied directly.
  ##   initial.y: The initial value of y.  This will be ignored if y is
  ##     provided, or if initial.state.prior is supplied directly.
  ## Returns:
  ##   state.specification, after appending the information necessary
  ##   to define a local level model

  if (missing(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))

  if (!missing(y)) {
    stopifnot(is.numeric(y))
    sdy <- sd(as.numeric(y), na.rm = TRUE)
    initial.y <- y[1]
  }

  if (missing(sigma.prior)) {
    ## The prior distribution says that sigma is small, and can be no
    ## larger than the sample standard deviation of the time series
    ## being modeled.
    sigma.prior <- SdPrior(.01 * sdy, upper.limit = sdy)
  }

  if (missing(initial.state.prior)) {
    initial.state.prior <- NormalPrior(initial.y, sdy)
  }

  level <- list(name = "trend",
                sigma.prior = sigma.prior,
                initial.state.prior = initial.state.prior,
                size = 1)
  class(level) <- c("LocalLevel", "StateModel")

  state.specification[[length(state.specification) + 1]] <- level
  return(state.specification)
}
