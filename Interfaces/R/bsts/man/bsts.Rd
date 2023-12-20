% Copyright 2011 Google Inc. All Rights Reserved.
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

\alias{bsts}
\name{bsts}
\title{Bayesian Structural Time Series}
\Rdversion{1.1}
\description{

  Uses MCMC to sample from the posterior distribution of a Bayesian
  structural time series model.  This function can be used either with
  or without contemporaneous predictor variables (in a time series
  regression).

  If predictor variables are present, the regression coefficients are
  fixed (as opposed to time varying, though time varying coefficients
  might be added as state component).  The predictors and response in
  the formula are contemporaneous, so if you want lags and differences
  you need to put them in the predictor matrix yourself.

  If no predictor variables are used, then the model is an ordinary
  state space time series model.

  The model allows for several useful extensions beyond standard
  Bayesian dynamic linear models.

  \itemize{

    \item{A spike-and-slab prior is used for the (static) regression
      component of models that include predictor variables.  This is
      especially useful with large numbers of regressor series.}

    \item{Both the spike-and-slab component (for static regressors) and
      the Kalman filter (for components of time series state) require
      observations and state variables to be Gaussian.  The \code{bsts}
      package allows for non-Gaussian error families in the observation
      equation (as well as some state components) by using data
      augmentation to express these families as conditionally
      Gaussian. }

    \item{As of version 0.7.0, \code{bsts} supports having multiple
      observations at the same time point.  In this case the basic model
      is taken to be

      \deqn{y_{t,j} = Z_t^T \alpha_t + \beta^Tx_{t, j} + \epsilon_{t,j}.}{%
	y[t, j] = Z'alpha[t] + beta'x[t, j] + epsilon[t, j].}

      This is a regression model where all observations with the same
      time point share a common time series effect.
    }
  }
}

\usage{
bsts(formula,
     state.specification,
     family = c("gaussian", "logit", "poisson", "student"),
     data,
     prior,
     contrasts = NULL,
     na.action = na.pass,
     niter,
     ping = niter / 10,
     model.options = BstsOptions(),
     timestamps = NULL,
     seed = NULL,
     ...)
}

\arguments{

  \item{formula}{ A formula describing the regression portion of the
    relationship between y and X.

    If no regressors are desired then the formula can be replaced by a
    numeric vector giving the time series to be modeled.  Missing values
    are not allowed in predictors, but they are allowed in the response
    variable.

    If the response variable is of class \code{\link[zoo]{zoo}},
    \code{\link[xts]{xts}}, or \code{\link{ts}}, then the time series
    information it contains will be used in many of the plotting methods
    called from \code{\link{plot.bsts}}.  }

  \item{state.specification}{A list with elements created by
    \code{\link{AddLocalLinearTrend}}, \code{\link{AddSeasonal}}, and similar
    functions for adding components of state.  See the help page for
    \code{\link{state.specification}}.}

  \item{family}{The model family for the observation equation.
    Non-Gaussian model families use data augmentation to recover a
    conditionally Gaussian model.}

  \item{data}{An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing the
    variables in the model.  If not found in \code{data}, the variables
    are taken from \code{environment(formula)}, typically the
    environment from which \code{\link{bsts}} is called.}


  \item{prior}{If regressors are supplied in the model formula, then
    this is a prior distribution for the regression component of the
    model, as created by \code{\link[BoomSpikeSlab]{SpikeSlabPrior}}.  The prior
    for the time series component of the model will be specified during
    the creation of state.specification.  This argument is only used if
    a formula is specified.

    If the model contains no regressors, then this is simply the prior
    on the residual standard deviation, expressed as an object created
    by \code{\link[Boom]{SdPrior}}. }

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

  \item{timestamps}{The timestamp associated with each value of the
    response.  This argument is primarily useful in cases where the
    response has missing gaps, or where there are multiple observations
    per time point.  If the response is a "regular" time series with a
    single observation per time point then you can leave this argument
    as \code{NULL}.  In that case, if either the response or the
    \code{data} argument is a type convertible to \code{\link{zoo}} then
    timestamps will be inferred.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{...}{ Extra arguments to be passed to
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
  If the model family is logit, then there are two ways one can
  format the response variable.  If the response is 0/1,
  TRUE/FALSE, or 1/-1, then the response variable can be passed
  as with any other model family.  If the response is a set of
  counts out of a specified number of trials then it can be
  passed as a two-column matrix, where the first column contains
  the counts of successes and the second contains the count of
  failures.

  Likewise, if the model family is Poisson, the response can be passed
  as a single vector of counts, under the assumption that each
  observation has unit exposure.  If the exposures differ across
  observations, then the resopnse can be a two column matrix, with the
  first column containing the event counts and the second containing
  exposure times.  }

\references{
  Scott and Varian (2014)
  "Predicting the Present with Bayesian Structural Time Series",
  International Journal of Mathematical Modelling and Numerical
  Optimization.  4--23.

  Scott and Varian (2015)
  "Bayesian Variable Selection for Nowcasting Economic Time Series",
  Economic Analysis of the Digital Economy, pp 119-135.

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
