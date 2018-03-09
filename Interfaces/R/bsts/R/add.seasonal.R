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

AddSeasonal <- function(state.specification,
                        y,
                        nseasons,
                        season.duration = 1,
                        sigma.prior = NULL,
                        initial.state.prior = NULL,
                        sdy) {
  ## Add a seasonal state component to state.specification
  ## Args:
  ##   state.specification: A list of state components.  If omitted,
  ##     an empty list is assumed.
  ##   y:  A numeric vector.  The time series to be modeled.
  ##   nseasons:  The number of seasons to model.
  ##   season.duration: The number of time periods each season is to
  ##     last.  For example.  If modeling a weekly effect on daily
  ##     data, then the weekly effect will last for season.duration =
  ##     7 days.
  ##   sigma.prior: An object created by SdPrior.  This is the prior
  ##     distribution on the standard deviation of the seasonal
  ##     increments.
  ##   initial.state.prior: An object created by NormalPrior.  The
  ##     prior distribution on the values of the initial state
  ##     (i.e. the state of the first observation).
  ##   sdy: The standard deviation of y.  This will be ignored if y is
  ##     provided, or if both sigma.prior and initial.state.prior are
  ##     supplied directly.
  ## Returns:
  ##   state.specification, after appending the information necessary
  ##   to define a seasonal state component.
  if (missing(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))
  if (!missing(y)) {
    stopifnot(is.numeric(y))
    sdy <- sd(as.numeric(y), na.rm = TRUE)
  }
  stopifnot(is.numeric(nseasons),
            length(nseasons) == 1,
            nseasons == round(nseasons))
  stopifnot(is.numeric(season.duration),
            length(season.duration) == 1,
            season.duration == round(season.duration))

  if (is.null(sigma.prior)) {
    ## The prior distribution says that sigma is small, and can be no
    ## larger than the sample standard deviation of the time series
    ## being modeled.
    sigma.prior <- SdPrior(.01 * sdy, upper.limit = sdy)
  }

  if (is.null(initial.state.prior)) {
    initial.state.prior <- NormalPrior(0, sdy)
  }

  seas <- list(name = paste("seasonal", nseasons, season.duration, sep="."),
               nseasons = nseasons,
               season.duration = season.duration,
               sigma.prior = sigma.prior,
               initial.state.prior = initial.state.prior,
               size = nseasons - 1)
  class(seas) <- c("Seasonal", "StateModel")
  state.specification[[length(state.specification) + 1]] <- seas
  return(state.specification)
}
