\name{plot.logit.spike}
\Rdversion{1.1}
\alias{plot.logit.spike}
\alias{plot.probit.spike}
\title{
  Plot a \code{\link{logit.spike}} object
}

\description{

  Plot a \code{\link{logit.spike}} object.  The default plot is a
  barplot of the marginal inclusion probabilities for each variable,
  as obtained by \code{\link{PlotMarginalInclusionProbabilities}}.
  See below for other types of plots.

}

\usage{
  \method{plot}{logit.spike}(x,
     y = c("inclusion", "coefficients", "scaled.coefficients", "fit",
           "residuals", "size", "help"),
     burn = SuggestBurnLogLikelihood(x$log.likelihood),
     ...)

  \method{plot}{probit.spike}(x,
     y = c("inclusion", "coefficients", "scaled.coefficients", "fit",
           "residuals", "size", "help"),
     burn = SuggestBurnLogLikelihood(x$log.likelihood),
     ...)
}
\arguments{
  \item{x}{
    An object of class \code{logit.spike}.
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

  The plot of the fit summary is handled by
  \code{\link{PlotLogitSpikeFitSummary}}.

  The plot of the residuals is handled by
  \code{\link{PlotLogitSpikeResiduals}}.

  The plot of model size is handled by \code{\link{PlotModelSize}}.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{PlotMarginalInclusionProbabilities}}
  \code{\link{PlotModelSize}}
  \code{\link{PlotLogitSpikeFitSummary}}
  \code{\link{PlotLogitSpikeResiduals}}
}

\examples{
## See the examples in ?logit.spike
}
\keyword{models}
\keyword{regression}
