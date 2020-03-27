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

AddTrig <- function(state.specification = NULL,
                    y,
                    period,
                    frequencies,
                    sigma.prior = NULL,
                    initial.state.prior = NULL,
                    sdy = sd(y, na.rm = TRUE),
                    method = c("harmonic", "direct")) {
  ## A trigonometric state model.
  ##
  ## Args:
  ##   state.specification: A list of state components.  If omitted, an empty
  ##     list is assumed.
  ##   y: A numeric vector.  The time series to be modeled.  This can be omitted
  ##     if sdy is provided.
  ##   period: A positive scalar giving the number of time steps required for
  ##     the longest cycle to repeat.
  ##   frequencies: A vector of positive real numbers giving the number of times
  ##     each cyclic component repeats in a period.  One sine and one cosine
  ##     term will be added for each frequency.
  ##   sigma.prior: The prior distribution for the standard deviations of the
  ##     changes in the sinusoid coefficients at each new time point.  This can
  ##     be NULL (in which case a default prior will be used), or a single
  ##     object of class SdPrior (which will be repeated for each sinusoid
  ##     independently).
  ##   initial.state.prior: The prior distribution for the values of
  ##     the sinusoid coefficients at time 0.  This can either be NULL
  ##     (in which case a default prior will be used), an object of
  ##     class MvnPrior.  If the prior is specified directly its
  ##     dimension must be twice the number of frequencies.
  ##   sdy: The standard deviation of the time series to be modeled.  This
  ##     argument is ignored if y is provided.
  if (is.null(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))

  if (!missing(y)) {
    stopifnot(is.numeric(y))
    sdy <- sd(as.numeric(y), na.rm = TRUE)
  } else if (missing(sdy)) {
    stop("At least one of y or sdy must be supplied to AddTrig.")
  }

  stopifnot(is.numeric(period),
            length(period) == 1,
            period > 0)
  stopifnot(is.numeric(frequencies),
            length(frequencies) > 0,
            all(frequencies > 0))
  method <- match.arg(method)

  ## Check the prior on the sinusoid coefficient increments.
  if (is.null(sigma.prior)) {
    sigma.prior <- SdPrior(0.01 * sdy, upper.limit = sdy)
  }
  stopifnot(inherits(sigma.prior, "SdPrior"))

  ## Check the prior on the initial state of the sinusoid coefficients.
  dimension <- 2 * length(frequencies)
  if (is.null(initial.state.prior)) {
    initial.state.prior <- MvnPrior(
        mean = rep(0, dimension),
        variance = diag(rep(sdy, dimension)^2))
  }
  stopifnot(inherits(initial.state.prior, "MvnDiagonalPrior") ||
            inherits(initial.state.prior, "MvnPrior"))
  stopifnot(length(initial.state.prior$mean) == dimension)

  ## All data has been checked and gathered at this point.  Return the
  ## object.
  trig <- list(name = paste0("trig.", period),
               frequencies = frequencies,
               period = period,
               sigma.prior = sigma.prior,
               initial.state.prior = initial.state.prior,
               size = dimension,
               method = method)
  class(trig) <- c("Trig", "StateModel")
  state.specification[[length(state.specification) + 1]] <- trig
  return(state.specification)
}
