% Copyright 2019 Steven L. Scott All Rights Reserved.
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

\alias{mbsts}
\name{mbsts}
\title{Multivariate Bayesian Structural Time Series}
\Rdversion{1.1}
\description{

  Fit a multivariate Bayesian structural time series model, also known
  as a "dynamic factor model."  

}

\usage{
mbsts(formula,
     state.specification,
     data,
     prior = NULL,
     contrasts = NULL,
     na.action = na.pass,
     niter,
     ping = niter / 10,
     seed = NULL,
     ...)
}

\arguments{

  \item{formula}{ A formula describing the regression portion of the
    relationship between y and X.

    If no regressors are desired then the formula can be replaced by a
    numeric matrix giving the multivariate time series to be modeled.

    If the response variable is of class \code{\link[zoo]{zoo}},
    \code{\link[xts]{xts}}, or \code{\link{ts}}, then the time series
    information it contains will be used in many of the plotting methods
    called from \code{\link{plot.bsts}}.
    
  }

  \item{state.specification}{A list with elements created by
    \code{\link{AddSharedLocalLevel}}, \code{\link{AddSeasonal}}, and
    similar functions for adding components of state.  }

  \item{data}{An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing the
    variables in the model.  If not found in \code{data}, the variables
    are taken from \code{environment(formula)}, typically the
    environment from which \code{\link{bsts}} is called.}

  \item{prior}{If regressors are supplied in the model formula, then
    this is a prior distribution for the regression component of the
    model, as created by
    \code{\link[BoomSpikeSlab]{MultivariateSpikeSlabPrior}}.  The prior
    for the time series component of the model will be specified during
    the creation of state.specification.  This argument is only used if
    a formula is specified.

    If the model contains no regressors, then this is simply the prior
    on the residual variance, expressed as an object created
    by \code{\link[Boom]{WishartPrior}}.
  }

  \item{contrasts}{An optional list containing the names of contrast
    functions to use when converting factors numeric variables in a
    regression formula.  This argument works exactly as it does in
    \code{\link{lm}}.  The names of the list elements correspond to
    factor variables in your model formula.  The list elements
    themselves are the names of contrast functions (see
    \code{help(\link[stats]{contr.treatment})} and the
    \code{contrasts.arg} argument to
    \code{\link{model.matrix.default}}).  This argument is only used if
    a model formula is specified, and even then the default is probably
    what you want.}

  \item{na.action}{What to do about missing values.  The default is to
    allow missing responses, but no missing predictors.  Set this to
    na.omit or na.exclude if you want to omit missing responses
    altogether.}

  \item{niter}{A positive integer giving the desired number of MCMC
    draws.}

  \item{ping}{ A scalar giving the desired frequency of status messages.
    If ping > 0 then the program will print a status message to the
    screen every \code{ping} MCMC iterations.}

  \item{model.options}{An object (list) returned by
    \code{\link{BstsOptions}}.  See that function for details.  }

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{...}{ Extra arguments to be passed to
    \code{\link[BoomSpikeSlab]{MultivariateSpikeSlabPrior}} (see the
    entry for the \code{prior} argument, above).}  }


\value{
  An object of class \code{\link{mbsts}} which is a list with the
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


\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.

  Goerge and McCulloch (1997)
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
  ## Example 1:  Time series (ts) data
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  pred <- predict(model, horizon = 12, burn = 100)
  par(mfrow = c(1,2))
  plot(model)
  plot(pred)

\dontrun{

  MakePlots <- function(model, ask = TRUE) {
    ## Make all the plots callable by plot.bsts.
    opar <- par(ask = ask)
    on.exit(par(opar))
    plot.types <- c("state", "components", "residuals",
                    "prediction.errors", "forecast.distribution")
    for (plot.type in plot.types) {
      plot(model, plot.type)
    }
    if (model$has.regression) {
      regression.plot.types <- c("coefficients", "predictors", "size")
      for (plot.type in regression.plot.types) {
        plot(model, plot.type)
      }
    }
  }

  ## Example 2: GOOG is the Google stock price, an xts series of daily
  ##            data.
  data(goog)
  ss <- AddSemilocalLinearTrend(list(), goog)
  model <- bsts(goog, state.specification = ss, niter = 500)
  MakePlots(model)

  ## Example 3:  Change GOOG to be zoo, and not xts.
  goog <- zoo(goog, index(goog))
  ss <- AddSemilocalLinearTrend(list(), goog)
  model <- bsts(goog, state.specification = ss, niter = 500)
  MakePlots(model)

  ## Example 4:  Naked numeric data works too
  y <- rnorm(100)
  ss <- AddLocalLinearTrend(list(), y)
  model <- bsts(y, state.specification = ss, niter = 500)
  MakePlots(model)

  ## Example 5:  zoo data with intra-day measurements
  y <- zoo(rnorm(100),
           seq(from = as.POSIXct("2012-01-01 7:00 EST"), len = 100, by = 100))
  ss <- AddLocalLinearTrend(list(), y)
  model <- bsts(y, state.specification = ss, niter = 500)
  MakePlots(model)

\dontrun{
  ## Example 6:  Including regressors
  data(iclaims)
  ss <- AddLocalLinearTrend(list(), initial.claims$iclaimsNSA)
  ss <- AddSeasonal(ss, initial.claims$iclaimsNSA, nseasons = 52)
  model <- bsts(iclaimsNSA ~ ., state.specification = ss, data =
                initial.claims, niter = 1000)
  plot(model)
  plot(model, "components")
  plot(model, "coefficients")
  plot(model, "predictors")
}
}

\dontrun{
  ## Example 7:  Regressors with multiple time stamps.
  number.of.time.points <- 50
  sample.size.per.time.point <- 10
  total.sample.size <- number.of.time.points * sample.size.per.time.point
  sigma.level <- .1
  sigma.obs <- 1

  ## Simulate some fake data with a local level state component.
  trend <- cumsum(rnorm(number.of.time.points, 0, sigma.level))
  predictors <- matrix(rnorm(total.sample.size * 2), ncol = 2)
  colnames(predictors) <- c("X1", "X2")
  coefficients <- c(-10, 10)
  regression <- as.numeric(predictors \%*\% coefficients)
  y.hat <- rep(trend, sample.size.per.time.point) + regression
  y <- rnorm(length(y.hat), y.hat, sigma.obs)

  ## Create some time stamps, with multiple observations per time stamp.
  first <- as.POSIXct("2013-03-24")
  dates <- seq(from = first, length = number.of.time.points, by = "month")
  timestamps <- rep(dates, sample.size.per.time.point)

  ## Run the model with a local level trend, and an unnecessary seasonal component.
  ss <- AddLocalLevel(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 7)
  model <- bsts(y ~ predictors, ss, niter = 250, timestamps = timestamps,
                seed = 8675309)
  plot(model)
  plot(model, "components")
}

## Example 8: Non-Gaussian data
## Poisson counts of shark attacks in Florida.
data(shark)
logshark <- log1p(shark$Attacks)
ss.level <- AddLocalLevel(list(), y = logshark)
model <- bsts(shark$Attacks, ss.level, niter = 500,
              ping = 250, family = "poisson", seed = 8675309)

## Poisson data can have an 'exposure' as the second column of a
## two-column matrix.
model <- bsts(cbind(shark$Attacks, shark$Population / 1000),
              state.specification = ss.level, niter = 500, 
              family = "poisson", ping = 250, seed = 8675309)

}
\keyword{models}
\keyword{regression}
