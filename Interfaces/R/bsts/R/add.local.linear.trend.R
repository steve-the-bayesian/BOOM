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

AddLocalLinearTrend <- function (state.specification = NULL,
                                 y,
                                 level.sigma.prior = NULL,
                                 slope.sigma.prior = NULL,
                                 initial.level.prior = NULL,
                                 initial.slope.prior = NULL,
                                 sdy,
                                 initial.y) {
  ## Adds a local linear trend component to the state model
  ## Args:
  ##   state.specification: A list of state components.  If omitted,
  ##     an empty list is assumed.
  ##   y:  A numeric vector.  The time series to be modeled.
  ##   level.sigma.prior: An object created by SdPrior.  The prior
  ##     distribution for the standard deviation of the increments in
  ##     the level component of state.
  ##   slope.sigma.prior: An object created by SdPrior.  The prior
  ##     distribution for the standard deviation of the increments in
  ##     the slope component of state.
  ##   initial.level.prior: An object created by NormalPrior.  The
  ##     prior distribution for the level component of state at the
  ##     time of the first observation.
  ##   initial.slope.prior: An object created by NormalPrior.  The
  ##     prior distribution for the slope component of state at the
  ##     time of the first observation.
  ##   sdy: The standard deviation of y.  This will be ignored if y is
  ##     provided, or if all four the required prior distributions are
  ##     supplied directly.
  ##   initial.y: The initial value of y.  This will be ignored if y is
  ##     provided, or if initial.level.prior is supplied directly.
  ## Returns:
  ##   state.specification, after appending the necessary information
  ##   to define a LocalLinearTrend model

  if (is.null(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))

  if (!missing(y)) {
    stopifnot(is.numeric(y))
    observed.y <- as.numeric(y[!is.na(y)])
    sdy <- sd(observed.y, na.rm = TRUE)
    initial.y <- observed.y[1]
  }

  if (is.null(level.sigma.prior)) {
    ## The prior distribution says that level.sigma is small, and can be no
    ## larger than the sample standard deviation of the time series
    ## being modeled.
    level.sigma.prior <- SdPrior(.01 * sdy, upper.limit = sdy)
  }

  if (is.null(slope.sigma.prior)) {
    ## The prior distribution says that slope.sigma is small, and can be no
    ## larger than the sample standard deviation of the time series
    ## being modeled.
    slope.sigma.prior <- SdPrior(.01 * sdy, upper.limit = sdy)
  }

  if (is.null(initial.level.prior)) {
    ## The mean of the initial level is the first observation.
    initial.level.prior <- NormalPrior(initial.y, sdy);
  }

  if (is.null(initial.slope.prior)) {
    if (!missing(y)) {
      ## If y is actually provided, then set the mean of the initial slope to
      ## the slope of a line connecting the first and last points of y.
      final.y <- tail(as.numeric(observed.y), 1)
      initial.slope.mean <- (final.y - initial.y) / length(y)
    } else {
      ## If y is missing set the mean of the initial slope to zero.
      initial.slope.mean <- 0
    }
    initial.slope.prior <- NormalPrior(initial.slope.mean, sdy);
  }

  llt <- list(name = "trend",
              level.sigma.prior = level.sigma.prior,
              slope.sigma.prior = slope.sigma.prior,
              initial.level.prior = initial.level.prior,
              initial.slope.prior = initial.slope.prior,
              size = 2)
  class(llt) <- c("LocalLinearTrend", "StateModel")

  state.specification[[length(state.specification) + 1]] <- llt
  return(state.specification)
}
