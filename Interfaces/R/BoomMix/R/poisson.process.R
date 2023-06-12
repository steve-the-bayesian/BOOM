# Copyright 2021 Steven L. Scott. All Rights Reserved.
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

HomogeneousPoissonProcess <- function(lambda.prior) {
  ## Creates an object of class HomogeneousPoissonProcess, whch
  ## inherits from the generic PoissonProcess.
  ## Args:
  ##   lambda.prior: An object of class GammaPrior specifying the
  ##     prior distribution on the Poisson rate parameter.  The first
  ##     argument to the GammaPrior is the prior number of observed
  ##     events.  The second is the prior exposure time (measured in days).
  ## Returns:
  ##   An object of class HomogeneousPoissonProcess, which inherits
  ##   from PoissonProcess.
  stopifnot(inherits(lambda.prior, "GammaPrior"))
  ans <- list(prior = lambda.prior)
  class(ans) <- c("HomogeneousPoissonProcess", "PoissonProcess")
  return(ans)
}

WeeklyCyclePoissonProcess <- function(average.daily.rate.prior,
                                      daily.dirichlet.prior,
                                      weekday.hourly.dirichlet.prior,
                                      weekend.hourly.dirichlet.prior) {
  ## Specifies a WeeklyCyclePoissonProcess object.
  ## Args:
  ##   average.daily.rate.prior: An object of class GammaPrior giving
  ##     the prior distribution on the average daily rate of the
  ##     process.
  ##   daily.dirichlet.prior: An object of class DirichletPrior giving
  ##     the prior distribution of the day of week effects.
  ##   weekday.hourly.dirichlet.prior: An object of class
  ##     DirichletPrior giving the prior distribution of the
  ##     hour-of-day effects for non-weekend days.
  ##   weekend.hourly.dirichlet.prior: An object of class
  ##     DirichletPrior giving the prior distribution of the
  ##     hour-of-day effects for weekend days.
  ## Returns:
  ##   An object of class WeeklyCyclePoissonProcess, which inherits
  ##   from PoissonProcess.
  stopifnot(inherits(average.daily.rate.prior, "GammaPrior"))
  stopifnot(inherits(daily.dirichlet.prior, "DirichletPrior"))
  stopifnot(inherits(weekday.hourly.dirichlet.prior, "DirichletPrior"))
  stopifnot(inherits(weekend.hourly.dirichlet.prior, "DirichletPrior"))
  stopifnot(length(daily.dirichlet.prior$prior.counts) == 7)
  stopifnot(length(weekday.hourly.dirichlet.prior) == 24)
  stopifnot(length(weekend.hourly.dirichlet.prior) == 24)

  ans <- list(average.daily.rate.prior = average.daily.rate.prior,
              daily.dirichlet.prior = daily.dirichlet.prior,
              weekday.hourly.dirichlet.prior = weekday.hourly.dirichlet.prior,
              weekend.hourly.dirichlet.prior = weekend.hourly.dirichlet.prior)
  class(ans) <- c("WeeklyCyclePoissonProcess", "PoissonProcess")
  return(ans)
}

SimulateWeeklyCyclePoissonProcess <-
  function(start, end, lambda, daily, weekday.hourly, weekend.hourly) {
  ## Simulates events from a Poisson process with a weekly cycle.
  ## Args:
  ##   start: A time convertible to POSIXct.  The beginning of the
  ##     observation window.
  ##   end: A time convertible to POSIXct.  The end of the observation
  ##     window.
  ##   lambda:  The average daily number of events.
  ##   daily: A vector of non-negative numbers that gives the each
  ##     day's multiplicative effect on the event rate.  The vector
  ##     must have 7 elements and sum to 7.
  ##   weekday.hourly: A vector of non-negative numbers giving each
  ##     hour's effect on the event rate.  The vector must have 24
  ##     elements and sum to 24.
  ##   weekend.hourly: A vector of non-negative numbers giving each
  ##     hour's effect on the event rate.  The vector must have 24
  ##     elements and sum to 24.
  ## Returns:
  ##   An object of class PointProcess with the specified weekly
  ##   profile.
  times <- as.POSIXct(c(start, end))
  start <- min(times)
  end <- max(times)

  duration.in.days <- as.numeric(difftime(end, start, "days"))
  max.rate <- lambda * max(outer(daily, weekend.hourly),
                           outer(daily, weekday.hourly))
  number.of.events <- rpois(1, max.rate * (duration.in.days))
  u <- sort(runif(number.of.events, 0, duration.in.days))
  u <- as.difftime(u, units = "days")
  cand <- as.POSIXlt(start + u)

  day <- weekdays(cand);
  day.names <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday")
  weekend <- day %in% c("Saturday", "Sunday")
  day <- factor(day, levels = day.names)
  hour <- cand$hour

  event.rate <- numeric(number.of.events)
  event.rate[weekend] <- (lambda * daily[day] *
                          weekend.hourly[1+hour])[weekend]
  event.rate[!weekend] <- (lambda * daily[day] *
                           weekend.hourly[1+hour])[!weekend]

  u <- runif(number.of.events)
  accept <- u < event.rate / max.rate
  ans <- PointProcess(cand[accept], start, end)
  return(ans)
}
