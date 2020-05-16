% Copyright 2018 Steven L. Scott. All Rights Reserved.
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

\alias{dirm}
\name{dirm}
\title{Dynamic intercept regression model}
\Rdversion{1.1}
\description{
  A dynamic intercept regression is a regression model where the
  intercept term is a state space model.  This model differs from
  \code{\link{bsts}} in that there can be multiple observations per time
  point.
}

\usage{
dirm(formula,
     state.specification,
     data,
     prior = NULL,
     contrasts = NULL,
     na.action = na.pass,
     niter,
     ping = niter / 10,
     model.options = DirmModelOptions(),
     timestamps = NULL,
     seed = NULL,
     ...)
}

\arguments{

  \item{formula}{ A formula, as you would supply to \code{\link{lm}}
    describing the regression portion of the relationship between y and
    X.}

  \item{state.specification}{A list with elements created by
    \code{\link{AddLocalLinearTrend}}, \code{\link{AddSeasonal}}, and similar
    functions for adding components of state.  See the help page for
    \code{\link{state.specification}}.

    The state specification describes the dynamic intercept term in the
    regression model.
  }

  \item{data}{An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing the
    variables in the model.  If not found in \code{data}, the variables
    are taken from \code{environment(formula)}, typically the
    environment from which \code{\link{dirm}} is called.}

  \item{prior}{A prior distribution for the regression component of the
    model, as created by \code{\link[BoomSpikeSlab]{SpikeSlabPrior}}.
    The prior for the time series component of the model will be
    specified during the creation of state.specification.  }

  \item{contrasts}{An optional list containing the names of contrast
    functions to use when converting factors numeric variables in a
    regression formula.  This argument works exactly as it does in
    \code{\link{lm}}.  The names of the list elements correspond to
    factor variables in your model formula.  The list elements
    themselves are the names of contrast functions (see
    \code{help(\link[stats]{contr.treatment})} and the
    \code{contrasts.arg} argument to
    \code{\link{model.matrix.default}}).  This argument can usually be
    omitted.}

  \item{na.action}{What to do about missing values.  The default is to
    allow missing responses, but no missing predictors.  Set this to
    na.omit or na.exclude if you want to omit missing responses
    altogether.}

  \item{niter}{A positive integer giving the desired number of MCMC
    draws.}

  \item{ping}{ A scalar giving the desired frequency of status messages.
    If ping > 0 then the program will print a status message to the
    screen every \code{ping} MCMC iterations.}

  \item{model.options}{ An object created by
    \code{\link{DirmModelOptions}} specifying the desired model
    options. }

  \item{timestamps}{The timestamp associated with each value of the
    response.  This is most likely a \code{\link{Date}} or
    \code{\link{POSIXt}}.  It is expected that there will be multiple
    observations per time point (otherwise 'bsts' should be used instead
    of 'dirm'), and thus the 'timestamps' argument will contain many
    duplicate values.
  }

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{\dots}{ Extra arguments to be passed to
    \code{\link[BoomSpikeSlab]{SpikeSlabPrior}} (see the entry for the
    \code{prior} argument, above).}
}

\value{
  An object of class \code{\link{bsts}} which is a list with the
  following components

  \item{coefficients}{ A \code{niter} by \code{ncol(X)} matrix of MCMC
    draws of the regression coefficients, where \code{X} is the design
    matrix implied by \code{formula}.  This is only present if a model
    formula was supplied.}

  \item{sigma.obs}{A vector of length \code{niter} containing MCMC draws
    of the residual standard deviation.}

  The returned object will also contain named elements holding the MCMC
  draws of model parameters belonging to the state models.  The names of
  each component are supplied by the entries in
  \code{state.specification}.  If a model parameter is a scalar, then
  the list element is a vector with \code{niter} elements.  If the
  parameter is a vector then the list element is a matrix with
  \code{niter} rows.  If the parameter is a matrix then the list element
  is a 3-way array with first dimension \code{niter}.

  Finally, if a model formula was supplied, then the returned object
  will contain the information necessary for the predict method to build
  the design matrix when a new prediction is made.
}

\details{

  The fitted model is a regression model with an intercept term given by
  a structural time series model.  This is similar to the model fit by
  \code{\link{bsts}}, but it allows for multiple observations per time
  period.

  Currently \code{dirm} only supports Gaussian observation errors, but
  look for that to change in future releases.

}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.

  George and McCulloch (1997)
  "Approaches for Bayesian variable selection", Statistica Sinica pp
  339--374.

}

\author{
  Steven L. Scott  \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{bsts}},
  \code{\link{AddLocalLevel}},
  \code{\link{AddLocalLinearTrend}},
  \code{\link{AddSemilocalLinearTrend}},
  \code{\link{AddSeasonal}}
  \code{\link{AddDynamicRegression}}
  \code{\link[BoomSpikeSlab]{SpikeSlabPrior}},
  \code{\link[Boom]{SdPrior}}.
}

\examples{

SimulateDirmData <- function(observation.sd = 1, trend.sd = .1,
                             time.dimension = 100, nobs.per.period = 3,
                             xdim = 4) {
  trend <- cumsum(rnorm(time.dimension, 0, trend.sd))
  total.sample.size <- nobs.per.period * time.dimension
  predictors <- matrix(rnorm(total.sample.size * xdim),
    nrow = total.sample.size)
  coefficients <- rnorm(xdim)
  expanded.trend <- rep(trend, each = nobs.per.period)
  response <- expanded.trend + predictors \%*\% coefficients + rnorm(
    total.sample.size, 0, observation.sd)
  timestamps <- seq.Date(from = as.Date("2008-01-01"),
                         len = time.dimension, by = "day")
  extended.timestamps <- rep(timestamps, each = nobs.per.period)
  return(list(response = response,
    predictors = predictors,
    timestamps = extended.timestamps,
    trend = trend,
    coefficients = coefficients))
}


data <- SimulateDirmData(time.dimension = 20)
ss <- AddLocalLevel(list(), data$response)

# In real life you'd want more than 50 MCMC iterations.
model <- dirm(data$response ~ data$predictors, ss, niter = 50,
  timestamps = data$timestamps)


}
\keyword{models}
\keyword{regression}
