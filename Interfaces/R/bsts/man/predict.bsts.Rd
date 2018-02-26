% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\alias{predict.bsts}
\alias{bsts.prediction}
\name{predict.bsts}
\title{Prediction for bayesian structural time series}
\Rdversion{1.1}
\description{Generated draws from the posterior predictive distribution
  of a \code{\link{bsts}} object.}
\usage{
\method{predict}{bsts}(object,
        newdata = NULL,
        horizon = 1,
        burn = SuggestBurn(.1, object),
        na.action = na.exclude,
        olddata = NULL,
        trials.or.exposure = 1,
        quantiles = c(.025, .975),
        ...)
}

\arguments{

  \item{object}{An object of class \code{bsts} created by a call to the
    function \code{\link{bsts}}.}

  \item{newdata}{a vector, matrix, or data frame containing the
    predictor variables to use in making the prediction.  This is only
    required if \code{object} contains a regression compoent.  If a data
    frame, it must include variables with the same names as the data
    used to fit \code{object}.  The first observation in \code{newdata}
    is assumed to be one time unit after the end of the last observation
    used in fitting \code{object}, and the subsequent observations are
    sequential time points.  If the regression part of \code{object}
    contains only a single predictor then \code{newdata} can be a
    vector.  If \code{newdata} is passed as a matrix it is the caller's
    responsibility to ensure that it contains the correct number of
    columns and that the columns correspond to those in
    \code{object$coefficients}.  }

  \item{horizon}{An integer specifying the number of periods into the
    future you wish to predict.  If \code{object} contains a regression
    component then the forecast horizon is \code{nrow(X)}, and this
    argument is not used.}

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
  Steven L. Scott \email{stevescott@google.com}
}

\seealso{
  \code{\link{bsts}}.
  \code{\link{AddLocalLevel}}.
  \code{\link{AddLocalLinearTrend}}.
  \code{\link{AddGeneralizedLocalLinearTrend}}.
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  pred <- predict(model, horizon = 12, burn = 100)
  plot(pred)
}
\keyword{models}
\keyword{regression}
