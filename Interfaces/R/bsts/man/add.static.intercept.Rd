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

\name{add.static.intercept}
\alias{AddStaticIntercept}

\Rdversion{1.1}
\title{
  Static Intercept State Component
}

\description{
Adds a static intercept term to a state space model.  If the model includes
a traditional trend component (e.g. local level, local linear trend, etc)
then a separate intercept is not needed (and will probably cause trouble,
as it will be confounded with the initial state of the trend model).
However, if there is no trend, or the trend is an AR process centered
around zero, then adding a static intercept will shift the center to a
data-determined value.
}

\usage{
AddStaticIntercept(
    state.specification,
    y,
    initial.state.prior = NormalPrior(y[1], sd(y, na.rm = TRUE)))
}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{y}{The time series to be modeled, as a numeric vector.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the intecept term.}
}

\value{ Returns a list with the information required to specify the
  state component. If initial.state.prior is specified then y is unused.
}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
  \code{\link[Boom]{NormalPrior}}
}

\keyword{models}
