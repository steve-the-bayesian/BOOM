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

\alias{predict.bsts}
\alias{bsts.prediction}
\name{predict.bsts}
\title{Prediction for Bayesian Structural Time Series}
\Rdversion{1.1}
\description{Generate draws from the posterior predictive distribution
  of a \code{\link{bsts}} object.}
\usage{
\method{predict}{bsts}(object,
        horizon = 1,
        newdata = NULL,
        timestamps = NULL,
        burn = SuggestBurn(.1, object),
        na.action = na.exclude,
        olddata = NULL,
        olddata.timestamps = NULL,
        trials.or.exposure = 1,
        quantiles = c(.025, .975),
        seed = NULL,
        ...)
}

\arguments{

  \item{object}{An object of class \code{bsts} created by a call to the
    function \code{\link{bsts}}.}

  \item{horizon}{An integer specifying the number of periods into the
    future you wish to predict.  If \code{object} contains a regression
    component then the forecast horizon is \code{nrow(X)}, and this
    argument is not used.}

  \item{newdata}{a vector, matrix, or data frame containing the
    predictor variables to use in making the prediction.  This is only
    required if \code{object} contains a regression component.  If a data
    frame, it must include variables with the same names as the data
    used to fit \code{object}.  The first observation in \code{newdata}
    is assumed to be one time unit after the end of the last observation
    used in fitting \code{object}, and the subsequent observations are
    sequential time points.  If the regression part of \code{object}
    contains only a single predictor then \code{newdata} can be a
    vector.  If \code{newdata} is passed as a matrix it is the caller's
    responsibility to ensure that it contains the correct number of
    columns and that the columns correspond to those in
    \code{object$coefficients}.
  }

  \item{timestamps}{ A vector of time stamps (of the same type as the
    timestamps used to fit \code{object}), with one per row of
    \code{newdata} (or element of \code{newdata}, if \code{newdata} is a
    vector).  The time stamps give the time points as which each
    prediction is desired.  They must be interpretable as integer (0 or
    larger) time steps following the last time stamp in \code{object}.
    If \code{NULL}, then the requested predictions are interpreted as
    being at 1, 2, 3, ... steps following the training data.  }

  \item{burn}{An integer describing the number of MCMC
    iterations in \code{object} to be discarded as burn-in.  If
    \code{burn <= 0} then no burn-in period will be discarded.
  }

  \item{na.action}{A function determining what should be done with
    missing values in \code{newdata}.}

  \item{olddata}{This is an optional component allowing predictions to
    be made conditional on data other than the data used to fit the
    model.  If omitted, then it is assumed that forecasts are to be made
    relative to the final observation in the training data.  If
    \code{olddata} is supplied then it will be filtered to get the
    distribution of the next state before a prediction is made, and it
    is assumed that the first entry in \code{newdata} comes immediately
    after the last entry in \code{olddata}.

    The value for \code{olddata} depends on whether or not \code{object}
    contains a regression component.

    \itemize{
      \item If a regression component is present, then \code{olddata}
      is a \code{data.frame} including variables with the same names
      as the data used to fit \code{object}, including the response .

      \item If no regression component is present, then \code{olddata}
      is a vector containing historical values of a time series.
    }
  }

  \item{olddata.timestamps}{ A set of timestamps corresponding to the
    observations supplied in \code{olddata}.  If \code{olddata} is
    \code{NULL} then this argument is not used.  If \code{olddata} is
    supplied and this is \code{NULL} then trivial timestamps (1, 2, ...)
    will be assumed.  Otherwise this argument behaves like the
    \code{timestamps} argument to the \code{\link{bsts}} function.  }

  \item{trials.or.exposure}{For logit or Poisson models, the number of
    binomial trials (or the exposure time) to assume at each time point
    in the forecast period.  This can either be a scalar (if the number
    of trials is to be the same for each time period), or it can be a
    vector with length equal to \code{horizon} (if the model contains no
    regression term) or \code{nrow(newdata)} if the model contains a
    regression term.}

  \item{quantiles}{A numeric vector of length 2 giving the lower and
    upper quantiles to use for the forecast interval estimate.
  }

  \item{seed}{An integer to use as the C++ random seed.  If \code{NULL}
  then the C++ seed will be set using the clock.}

  \item{\dots}{This is a dummy argument included to match the signature
    of the generic \code{\link{predict}} function.  It is not used.}

}

\details{

  Samples from the posterior distribution of a Bayesian structural time
  series model.  This function can be used either with or without
  contemporaneous predictor variables (in a time series regression).

  If predictor variables are present, the regression coefficients are
  fixed (as opposed to time varying, though time varying coefficients
  might be added as state component).  The predictors and response in
  the formula are contemporaneous, so if you want lags and differences
  you need to put them in the predictor matrix yourself.

  If no predictor variables are used, then the model is an ordinary
  state space time series model.
}



\value{
  Returns an object of class \code{bsts.prediction}, which is a list
  with the following components.

  \item{mean}{A vector giving the posterior mean of the prediction.}
  \item{interval}{A two (column/row?) matrix giving the upper and lower
  bounds of the 95 percent credible interval for the prediction.}
  \item{distribution}{A matrix of draws from the posterior predictive
  distribution.  Each row in the matrix is one MCMC draw.  Columns
  represent time.}
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
  \code{\link{AddLocalLevel}}.
  \code{\link{AddLocalLinearTrend}}.
  \code{\link{AddSemilocalLinearTrend}}.
}

\examples{

# The number of MCMC draws in the following examples is artificially low.


  ## Making predictions when there is no regression component.
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 250)
  pred <- predict(model, horizon = 12, burn = 100)
  plot(pred)

  ## An example using the olddata argument.
  full.pred <- pred
  training <- window(y, end = c(1959, 12))
  model <- bsts(training, state.specification = ss, niter = 250)
  ## Predict the next 12 months.
  pred <- predict(model, horizon = 12)
  ## Compare the predictions to the actual data.
  plot(pred)
  lines(as.numeric(y, col = "red", lty = 2, lwd = 2))

  ## Predict the 12 months of 1961 based on the posterior distribution
  ## of the model fit to data through 1959, but with state filtered
  ## through 1960.
  updated.pred <- predict(model, horizon = 12, olddata = y)
  par(mfrow = c(1, 2))
  plot(full.pred, ylim = c(4, 7))
  plot(updated.pred, ylim = c(4, 7))

  ## Examples including a regression component.
  ##
  data(iclaims)
  training <- initial.claims[1:402, ]
  holdout1 <- initial.claims[403:450, ]
  holdout2 <- initial.claims[451:456, ]

\dontrun{

## This example puts the total run time over 5 seconds, which is a CRAN
## violation.

  ss <- AddLocalLinearTrend(list(), training$iclaimsNSA)
  ss <- AddSeasonal(ss, training$iclaimsNSA, nseasons = 52)
  ## In real life you'd want more iterations...
  model <- bsts(iclaimsNSA ~ ., state.specification = ss, data =
                training, niter = 100)

  ## Predict the holdout set given the training set.
  ## This is really fast, because we can use saved state from the MCMC
  ## algorithm.
  pred.full <- predict(model, newdata = rbind(holdout1, holdout2))

  ## Predict holdout 2, given training and holdout1.
  ## This is much slower because we need to re-filter the 'olddata' before
  ## simulating the predictions.
  pred.update <- predict(model, newdata = holdout2,
    olddata = rbind(training, holdout1))
}
}
\keyword{models}
\keyword{regression}
