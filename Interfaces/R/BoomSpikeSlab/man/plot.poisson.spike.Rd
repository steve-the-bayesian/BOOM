\name{plot.poisson.spike}
\Rdversion{1.1}
\alias{plot.poisson.spike}
\title{
  Plot a \code{\link{poisson.spike}} object
}

\description{

  Plot a \code{\link{poisson.spike}} object.  The default plot is a
  barplot of the marginal inclusion probabilities for each variable,
  as obtained by \code{\link{PlotMarginalInclusionProbabilities}}.
  See below for other types of plots.

}

\usage{
  \method{plot}{poisson.spike}(x,
     y = c("inclusion", "coefficients", "scaled.coefficients", "size", "help"),
     burn = SuggestBurnLogLikelihood(x$log.likelihood),
     ...)
}
\arguments{
  \item{x}{
    An object of class \code{poisson.spike}.
  }

  \item{y}{
    The type of plot desired.
  }

  \item{burn}{
    The number of MCMC iterations to discard as burn-in.
  }

  \item{\dots}{ Additional arguments passed to the specific functions
    that do the plotting.  }

}

\details{

  The default plot is a barplot showing the marginal inclusion
  probabilities of the coefficients, constructed using
  \code{\link{PlotMarginalInclusionProbabilities}}.

  The plot of model size is handled by \code{\link{PlotModelSize}}.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{PlotMarginalInclusionProbabilities}}
  \code{\link{PlotModelSize}}
}

\examples{
## See the examples in ?poisson.spike
}
\keyword{models}
\keyword{regression}
