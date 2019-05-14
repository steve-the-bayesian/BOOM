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

  ** NOTE ** This code is experimental.  Please feel free to experiment
  with it and report any bugs to the maintainer.  Expect it to improve
  substantially in the next release.  }

\usage{
mbsts(formula,
      shared.state.specification,
      series.state.specification = NULL,
      data = NULL,
      timestamps = NULL,
      series.id = NULL,
      prior = NULL,  # TODO
      opts = NULL,
      contrasts = NULL,
      na.action = na.pass,
      niter,
      ping = niter / 10,
      data.format = c("long", "wide"),
      seed = NULL,
      ...) 
}

\arguments{

  \item{formula}{A formula describing the regression portion of the
    relationship between y and X.

    If no regressors are desired then the formula can be replaced by a
    numeric matrix giving the multivariate time series to be modeled.
  }

  \item{shared.state.specification}{A list with elements created by
    \code{\link{AddSharedLocalLevel}}, and similar functions for adding
    components of state.

    This list defines the components of state which are shared across
    all time series.  These are the "factors" in the dynamic factor
    model.
  }

  \item{series.state.specification}{ This argument specifies state
    components needed by a particular series.  Not all series need have
    the same state components (e.g. some series may require a seasonal
    component, while others do not).  It can be \code{NULL}, indicating
    that there are no series-specific state components.

    It can be a list of elements created by \code{\link{AddLocalLevel}},
    \code{\link{AddSeasonal}}, and similar functions for adding state
    component to scalar bsts models.  In this case the same,
    independent, individual components will be added to each series.
    For example, each series will get its own independent Seasonal state
    component if AddSeasonal was used to add a seasonal component to
    this argument.

    In its most general form, this argument can be a list of lists, some
    of which can be NULL, but with non-NULL lists specifying state
    components for individual series, as above.
  }

  \item{data}{An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing
    the variables mentioned in the \code{formula} argument.  If not
    found in \code{data}, the variables are taken from
    \code{environment(formula)}, typically the environment from which
    \code{\link{bsts}} is called.}


  \item{timestamps}{A vector of timestamps indicating the time of each
    observation.  If \code{data.format} is \code{"long"} then this
    argument is required.  If "wide" data is passed then it is optional.

    TODO:  TEST THIS under wide and long formats in regression and
    non-regression settings.
  }

  \item{series.id}{A factor (or object coercible to factor) indicating
    the series to which each observation in "long" format belongs.  This
    argument is ignored for data in "wide" format.  }
  
  \item{prior}{A list of \code{\link{SpikeSlabPrior}} objects, one for
    each time series.  Or this argument can be NULL in which case a
    default prior will be used.  Note that the prior is on both the
    regression coefficients and the residual sd for each time series. }

  \item{opts}{A list containing model options.  This is currently only
    used for debugging, so leave this as \code{NULL}.}
  
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


  \item{data.format}{Whether the data are store in wide (each row is a
    time point, and columns are values from different series) or long
    (each row is the value of a particular series at a particular point
    in time) format.  For \code{"long"} see \code{timestamps} and
    \code{series.id}.
  }
  
  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{...}{ Extra arguments to be passed to
    \code{\link[BoomSpikeSlab]{SpikeSlabPrior}} (see the
    entry for the \code{prior} argument, above).}
}


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
seed <- 8675309
set.seed(seed)

ntimes <- 250
nseries <- 20
nfactors <- 6
residual.sd <- 1.2
state.innovation.sd <- .75

##---------------------------------------------------------------------------
## simulate latent state for fake data.
##---------------------------------------------------------------------------
state <- matrix(rnorm(ntimes * nfactors, 0, state.innovation.sd), nrow = ntimes)
for (i in 1:ncol(state)) {
  state[, i] <- cumsum(state[, i])
}

##---------------------------------------------------------------------------
## Simulate "observed" data from state.
##---------------------------------------------------------------------------
observation.coefficients <- matrix(rnorm(nseries * nfactors), nrow = nseries)
print("observation.coefficients =\n", observation.coefficients)
diag(observation.coefficients) <- 1.0
observation.coefficients[upper.tri(observation.coefficients)] <- 0

errors <- matrix(rnorm(nseries * ntimes, 0, residual.sd), ncol = ntimes)
y <- t(observation.coefficients \%*\% t(state) + errors)

\dontrun{
cat("simulated y (first 10 rows): \n")
print(y[1:10, ])

##---------------------------------------------------------------------------
## Plot the data.
##---------------------------------------------------------------------------
par(mfrow=c(1,2))
plot.ts(y, plot.type="single", col = rainbow(nseries), main = "observed data")
plot.ts(state, plot.type = "single", col = 1:nfactors, main = "latent state")
}

##---------------------------------------------------------------------------
## Fit the model
##---------------------------------------------------------------------------
ss <- AddSharedLocalLevel(list(), y, nfactors = nfactors)

opts <- list("fixed.state" = t(state),
  fixed.residual.sd = rep(residual.sd, nseries),
  fixed.regression.coefficients = matrix(rep(0, nseries), ncol = 1))

model <- mbsts(y, shared.state.specification = ss, niter = 100,
  data.format = "wide", seed = seed)

##---------------------------------------------------------------------------
## Plot the state
##---------------------------------------------------------------------------
par(mfrow=c(1, nfactors))
ylim <- range(model$shared.state, state)
for (j in 1:nfactors) {
  PlotDynamicDistribution(model$shared.state[, j, ], ylim=ylim)
  lines(state[, j], col = "blue")
}

##---------------------------------------------------------------------------
## Plot the factor loadings.
##---------------------------------------------------------------------------
\dontrun{
opar <- par(mfrow=c(nfactors,1), mar=c(0, 4, 0, 4), omi=rep(.25, 4))
burn <- 10
for(j in 1:nfactors) {
  BoxplotTrue(model$shared.local.level.coefficients[-(1:burn), j, ],
    t(observation.coefficients[, j]), axes=F, truth.color="blue")
  abline(h=0, lty=3)
  box()
  axis(2)
}
axis(1)
par(opar)
}

##---------------------------------------------------------------------------
## Plot the predicted values of the series.
##---------------------------------------------------------------------------
index <- 1:12
nr <- floor(sqrt(length(index)))
nc <- ceiling(length(index) / nr)
opar <- par(mfrow = c(nr, nc), mar = c(2, 4, 1, 2))
for (i in index) {
  PlotDynamicDistribution(
    model$shared.state.contributions[, 1, i, ]
    + model$regression.coefficients[, i, 1]
  , ylim=range(y))
  points(y[, i], col="blue", pch = ".", cex = .2)
}
par(opar)
}
\keyword{models}
\keyword{regression}
