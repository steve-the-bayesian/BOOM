% Copyright 2018 Google Inc. All Rights Reserved.
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public
% License along with this library; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

\name{add.monthly.annual.cycle}
\alias{AddMonthlyAnnualCycle}
\alias{MonthlyAnnualCycle}

\Rdversion{1.1}
\title{Monthly Annual Cycle State Component}

\description{A seasonal state component for daily data, representing
  the contribution of each month to the annual seasonal cycle.
  I.e. this is the "January, February, March, ..." effect, with 12
  seasons.  There is a step change at the start of each month, and then
  the contribution of that month is constant over the course of the
  month.

  Note that if you have anything other than daily data, then you're
  probably looking for \code{\link{AddSeasonal}} instead.

  The state of this model is an 11-vector \eqn{\gamma_t}{gamma[t, ]}
  where the first element is the contribution to the mean for the
  current month, and the remaining elements are the values for the 10
  most recent months.  When \eqn{t}{t} is the first day in the month
  then

  \deqn{\gamma_{t+1} = -\sum_{i = 2}^11 \gamma_{t, i} + %
    \epsilon_t  \qquad \epsilon_t \sim \mathcal{N}(0, \sigma)}{%
    gamma[t+1, 1] = -1 * sum(gamma[t, -1]) + rnorm(1, 0, sigma) }

  And the remaining elements are \eqn{\gamma_t}{gamma[t, ]} shifted down
  one.  When \eqn{t}{t} is any other day then \eqn{\gamma_{t+1} = %
  \gamma_t}{gamma[t+1, ] == gamma[t, ]}.

}

\usage{
AddMonthlyAnnualCycle(state.specification,
                      y,
                      date.of.first.observation = NULL,
                      sigma.prior = NULL,
                      initial.state.prior = NULL,
                      sdy)
}

\arguments{

  \item{state.specification}{A list of state components, to which the
    monthly annual cycle will be added.  If omitted, an empty list will
    be assumed.}

  \item{y}{ The time series to be modeled, as a numeric vector.}

  \item{date.of.first.observation}{The time stamp of the first
    observation in \code{y}, as a \code{\link{Date}} or
    \code{\link{POSIXt}} object.  If \code{y} is a \code{\link{zoo}}
    object with appropriate time stamps then this argument will be
    deduced.}

  \item{sigma.prior}{An object created by \code{\link[Boom]{SdPrior}}
    describing the prior distribution for the standard deviation of the
    random walk increments.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the the initial state vector (at time 1).}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }
}

\examples{

  ## Let's simulate some fake daily data with a monthly cycle.
\dontrun{
  residuals <- rnorm(365 * 5)
}
\dontshow{
  residuals <- rnorm(365 * 2)
}
  n <- length(residuals)
  dates <- seq.Date(from = as.Date("2014-01-01"),
                    len = n,
                    by = 1)
  monthly.cycle <- rnorm(12)
  monthly.cycle <- monthly.cycle - mean(monthly.cycle)
  timestamps <- as.POSIXlt(dates)
  month <- timestamps$mon + 1

  new.month <- c(TRUE, diff(timestamps$mon) != 0)
  month.effect <- cumsum(new.month) %% 12
  month.effect[month.effect == 0] <- 12

  response <- monthly.cycle[month] + residuals
  response <- zoo(response, timestamps)

  ## Now let's fit a bsts model to the daily data with a monthly annual
  ## cycle.
  ss <- AddLocalLevel(list(), response)
  ss <- AddMonthlyAnnualCycle(ss, response)

  ## In real life you'll probably want more iterations.
  model <- bsts(response, state.specification = ss, niter = 250)
  plot(model)
  plot(model, "monthly")
}
