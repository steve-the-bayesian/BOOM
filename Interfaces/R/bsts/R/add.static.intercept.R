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

AddStaticIntercept <- function(
    state.specification,
    y,
    initial.state.prior = NormalPrior(y[1], sd(y, na.rm = TRUE))) {
  ## Adds a static intercept term to a state space model.  If the model includes
  ## a traditional trend component (e.g. local level, local linear trend, etc)
  ## then a separate intercept is not needed (and will probably cause trouble,
  ## as it will be confounded with the initial state of the trend model).
  ## However, if there is no trend, or the trend is an AR process centered
  ## around zero, then adding a static intercept will shift the center to a
  ## data-determined value.
  ##
  ## Args:
  ##   state.specification: A list of state components.  If omitted, an empty
  ##     list is assumed.
  ##   y:  A numeric vector.  The time series to be modeled.
  ##   initial.state.prior: An object of class NormalPrior.  The prior
  ##     distribution on the values of the initial state (i.e. the state of the
  ##     first observation).
  ## Returns:
  ##   state.specification, after appending the information necessary
  ##   to define a static intercept term.
  if (missing(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))
  stopifnot(inherits(initial.state.prior, "NormalPrior"))
  component <- list(name = "Intercept",
                    initial.state.prior = initial.state.prior,
                    size = 1)
  class(component) <- c("StaticIntercept", "StateModel")
  state.specification[[length(state.specification) + 1]] <- component
  return(state.specification)
}
