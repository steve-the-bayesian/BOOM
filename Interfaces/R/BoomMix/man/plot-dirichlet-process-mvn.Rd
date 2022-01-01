\name{plot-dirichlet-process-mvn}

\alias{plot.DirichletProcessMvn}
\alias{PlotDpMvnMeans}
\alias{PlotDpMvnMeanPairs}
\alias{PlotDpMvnNclusters}
\alias{PlotDpMvnLoglike}

\title{Plot Method for DP Mixtures of Multivariate Normals}
\Rdversion{1.0}
\description{
  Plot a \code{\link{DirichletProcessMvn}} object.
}

\usage{
\method{plot}{DirichletProcessMvn}(
   x,
   y = c("means", "nclusters", "pairs", "log.likelihood", "help"),
   ...)

PlotDpMvnMeanPairs(model, nclusters, burn = NULL, dims = NULL, gap = 0, ...)
PlotDpMvnMeans(model, nclusters, burn = NULL, dims = NULL, ...)
PlotDpMvnNclusters(object, burn = NULL, ...)
PlotDpMvnLoglike(object, burn = 0, xlab = "Iteration", ylab =
"Log Likelihood", ...)
}

\arguments{

  \item{x, model, object}{The \code{\link{DirichletProcessMvn}} object to be plotted.}

  \item{y}{A string giving the type of plot desired.}

  \item{burn}{The number of MCMC iterations to discard as burn-in. If
    \code{NULL} a burn-in value will be used based on the log likelihood.}

  \item{nclusters}{Plot the value for the subset of draws associated
    with \code{nclusters} clusters.}

  \item{dims}{The subset of dimensions to plot.  The default is to plot
    everything.}

  \item{gap}{The amount of space between frames.  Expressed in terms of
    lines of text. }

  \item{xlab}{Label for the horizontal axis.}

  \item{ylab}{Label for the vertical axis.}

  \item{...}{Additional arguments to specific plotting functions.}

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomMix-package}}.
}

\keyword{models}
