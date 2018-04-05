\name{diagnostic-plots}
\title{Diagnostic Plots}

\alias{qqdist}
\alias{AcfDist}

\description{
  Diagnostic plots for distributions of residuals.
}

\usage{
  qqdist(draws, \dots)
  AcfDist(draws, lag.max = NULL, xlab = "Lag", ylab = "Autocorrelation", \dots)
}

\arguments{

  \item{draws}{A matrix of Monte Carlo draws of residual errors.  Each
    row is a Monte Carlo draw, and each column is an observation.  In the
    case of AcfDist successive observations are assumed to be sequential
    in time.}

  \item{lag.max}{The number of lags to plot in the autocorrelation
    function.  See \code{\link{acf}}.}

  \item{xlab}{Label for the horizontal axis.}
  \item{ylab}{Label for the vertical axis.}

  \item{\dots}{Extra arguments passed to either \code{\link{boxplot}}
    (for \code{AcfDist}) or \code{\link[Boom]{PlotDynamicDistribution}} (for
    \code{qqdist}).}

}

\details{

  \code{qqdist} sorts the columns of \code{draws} by their mean, and
  plots the resulting set of curves against the quantiles of the
  standard normal distribution.  A reference line is added, and the mean
  of each column of draws is represented by a blue dot.  The dots and
  the line are the transpose of what you get with \code{\link{qqnorm}}
  and \code{\link{qqline}}.

  \code{AcfDist} plots the posterior distribution of the autocorrelation
  function using a set of side-by-side boxplots.

}


\examples{

data(AirPassengers)
y <- log(AirPassengers)

ss <- AddLocalLinearTrend(list(), y)
ss <- AddSeasonal(ss, y, nseasons = 12)
model <- bsts(y, ss, niter = 500)

r <- residuals(model)
par(mfrow = c(1,2))
qqdist(r)   ## A bit of departure in the upper tail
AcfDist(r)  

}
