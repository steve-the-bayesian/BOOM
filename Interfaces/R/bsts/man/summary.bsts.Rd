% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{summary.bsts}
\title{Summarize a Bayesian structural time series object}

\alias{summary.bsts}

\description{Print a summary of a \code{\link{bsts}} object.}

\usage{
  \method{summary}{bsts}(object, burn = SuggestBurn(.1, object), \dots)
}

\arguments{
  \item{object}{An object of class \code{\link{bsts}} created by the function
    of the same name.}

  \item{burn}{The number of MCMC iterations to discard as burn-in.}

  \item{\dots}{Additional arguments passed to
    \code{\link[BoomSpikeSlab]{summary.lm.spike}} if \code{object} has a regression component.}
}

\value{
  Returns a list with the following elements.
  \item{residual.sd}{The posterior mean of the residual standard
      deviation parameter.}

    \item{prediction.sd}{The standard deviation of the one-step-ahead
      prediction errors for the training data.}

    \item{rsquare}{Proportion by which the residual variance is less
      than the variance of the original observations.}

     \item{relative.gof}{Harvey's goodness of fit statistic.  Let
       \eqn{\nu}{nu} denote the one step ahead prediction errors,
       \eqn{n}{n} denote the length of the series, and \eqn{y}{y} denote
       the original series.  The goodness of fit statistic is \deqn{ 1 -
       \sum_{i = 1}^n \nu_i^2 / \sum_{i = 2}{n} (\Delta y_i- \Delta \bar
       y)^2.}{ 1 - sum(nu^2) / (n-2) * var(diff(y)).}

     This statistic is analogous to \eqn{R^2}{rsquare} in a regression
     model, but the reduction in sum of squared errors is relative to a
     random walk with a constant drift, \deqn{y_{t+1} = y_t + \beta +
     \epsilon_t,}{y[t+1] = y[t] + beta + epsilon[t],} which Harvey
     (1989, equation 5.5.14) argues is a more relevant baseline than a
     simple mean.  Unlike a traditional R-square statistic, this can be
     negative.}

   \item{size}{Distribution of the number of nonzero coefficients
       appearing in the model}

  \item{coefficients}{If \code{object} contains a regression component then the
    output contains matrix with rows corresponding to coefficients, and
    columns corresponding to:
    \itemize{
      \item The posterior probability the variable is included.
      \item The posterior probability that the variable is positive.
      \item The conditional expectation of the coefficient, given inclusion.
      \item The conditional standard deviation of the coefficient, given inclusion.
    }
  }
}

\references{Harvey's goodness of fit statistic is from Harvey (1989)
  \emph{Forecasting, structural time series models, and the Kalman filter.}
  Page 268.}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 100)
  summary(model, burn  = 20)
}

\seealso{
  \code{\link{bsts}}, \code{\link{plot.bsts}}, \code{\link[BoomSpikeSlab]{summary.lm.spike}}
}
