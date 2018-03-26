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

AddMonthlyAnnualCycle <- function(state.specification,
                                  y,
                                  date.of.first.observation = NULL,
                                  sigma.prior = NULL,
                                  initial.state.prior = NULL,
                                  sdy) {
  ## Adds a monthly 12-cycle that updates at the first of each month.  Intended
  ## for daily data.
  ##
  ## Args:
  ##   state.specification: A list of state components.  If omitted,
  ##     an empty list is assumed.
  ##   y:  A numeric vector.  The time series to be modeled.
  ##   date.of.first.observation: An object of type 'Date' or 'POSIXt' giving
  ##     the date of the first observation in 'y'.  If 'y' is of type 'zoo' and
  ##     the index of 'y' is Date or POSIXt then a NULL value here signals that
  ##     the date of the first observation should be inferred.
  ##   sigma.prior: An object created by SdPrior.  This is the prior
  ##     distribution describing the standard deviation of the seasonal
  ##     increments.
  ##   initial.state.prior: An object created by NormalPrior.  The prior
  ##     distribution on the values of the initial state (i.e. the state of the
  ##     first observation).
  ##   sdy: The standard deviation of y.  This will be ignored if y is provided,
  ##     or if both sigma.prior and initial.state.prior are supplied directly.
  ##
  ## Returns:
  ##   state.specification, after appending an element for the monthly
  ##   component.
  if (missing(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))
  if (!missing(y)) {
    stopifnot(is.numeric(y))
    sdy <- sd(as.numeric(y), na.rm = TRUE)
  }

  if (is.null(sigma.prior)) {
    ## The default prior says that sigma is small, and can be no larger than the
    ## sample standard deviation of the time series being modeled.
    sigma.prior <- SdPrior(.01 * sdy, upper.limit = sdy)
  }
  stopifnot(inherits(sigma.prior, "SdPrior"))

  if (is.null(initial.state.prior)) {
    initial.state.prior <- NormalPrior(0, sdy)
  }
  stopifnot(inherits(initial.state.prior, "NormalPrior"))

  if (is.null(date.of.first.observation)) {
    if (is.zoo(y)) {
      dates <- index(y)
      date.of.first.observation <- as.Date(dates[1])
    }
  }
  stopifnot(inherits(date.of.first.observation, c("Date", "POSIXt")),
            length(date.of.first.observation) == 1)
  date.of.first.observation <- as.POSIXlt(date.of.first.observation)

  monthly <- list(
      name = "Monthly",
      sigma.prior = sigma.prior,
      initial.state.prior = initial.state.prior,
      first.observation.month = as.integer(
          date.of.first.observation$mon + 1),
      first.observation.day = as.integer(
          date.of.first.observation$mday),
      first.observation.year = as.integer(
          1900 + date.of.first.observation$year),
      size = 11)
  class(monthly) <- c("Monthly", "StateModel")
  state.specification[[length(state.specification) + 1]] <- monthly
  return(state.specification)
}
