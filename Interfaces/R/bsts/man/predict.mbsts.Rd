% Copyright 2019 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{predict.mbsts}
\alias{predict.mbsts}
\alias{mbsts.prediction}
\title{Prediction for Multivariate Bayesian Structural Time Series}
\Rdversion{1.1}
\description{Generate draws from the posterior predictive distribution
  of an \code{\link{mbsts}} object.}

\usage{
\method{predict}{mbsts}(object,
        horizon = 1,
        newdata = NULL,
        timestamps = NULL,
        burn = SuggestBurn(.1, object),
        na.action = na.exclude,
        quantiles = c(.025, .975),
        seed = NULL,
        ...)
}

\arguments{

  \item{object}{An object of class \code{\link{mbsts}}.}

  \item{horizon}{An integer specifying the number of periods into the
    future you wish to predict.  If \code{object} contains a regression
    component then the forecast horizon is \code{nrow(newdata)} and this
    argument is not used.}

  \item{newdata}{A vector, matrix, or data frame containing the
    predictor variables to use in making the prediction.  This is only
    required if \code{object} contains a regression component.  If a
    data frame, it must include variables with the same names as the
    data used to fit \code{object}.  The first observation in newdata is
    assumed to be one time unit after the end of the last data used in
    fitting \code{object}, and the subsequent observations are
    sequential time points.  If the regression part of \code{object}
    contains only a single predictor then newdata can be a vector.  If
    \code{newdata} is passed as a matrix it is the caller's
    responsibility to ensure that it contains the correct number of
    columns and that the columns correspond to those in
    object$coefficients.  }

  \item{timestamps}{A vector of time stamps (of the same type as the
    timestamps used to fit \code{object}), with one per row of
    \code{newdata} (or element of \code{newdata}, if \code{newdata} is a
    vector).  The time stamps give the time points as which each
    prediction is desired.  They must be interpretable as integer (0 or
    larger) time steps following the last time stamp in \code{object}.
    If \code{NULL}, then the requested predictions are interpreted as
    being at 1, 2, 3, ... steps following the training data.}

  \item{burn}{An integer describing the number of MCMC iterations in
    \code{object} to be discarded as burn-in.  If burn <= 0 then no
    burn-in period will be discarded.}

  \item{na.action}{A function determining what should be done with
    missing values in \code{newdata}.}

  \item{quantiles}{A numeric vector of length 2 giving the lower and
    upper quantiles to use for the forecast interval estimate.}

  \item{seed}{An integer to use as the C++ random seed.  If
    \code{NULL} then the C++ seed will be set using the clock.}

  \item{\dots}{Not used.  Present to match the signature of the
    default predict method.}
}

\details{
  The prediction is based off of samples taken from the posterior
  distribution of a multivariate Bayesian structural time series model.

  As an added convenience, means and interval estimates are produced
  from the posterior predictive distribution.
}

\value{
  Returns an object of class mbsts.prediction, which is a list.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{mbsts}}.
  \code{\link{predict.bsts}}
  \code{\link{plot.mbsts.prediction}}
}
