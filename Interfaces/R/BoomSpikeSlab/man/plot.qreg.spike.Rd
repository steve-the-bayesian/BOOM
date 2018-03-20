\name{plot.qreg.spike}
\Rdversion{1.1}
\alias{plot.qreg.spike}
\title{
  Plot the results of a spike and slab regression.
}
\description{

  The default plot is a barplot of the marginal inclusion probabilities
  for each variable, as obtained by
  \code{\link{PlotMarginalInclusionProbabilities}}.  Other interesting
  plots can be obtained by supplying a string as the second argument.

}

\usage{
  \method{plot}{qreg.spike}(x,
     y = c("inclusion", "coefficients", "scaled.coefficients",
              "size", "help"),
     burn = SuggestBurnLogLikelihood(x$log.likelihood),
     ...)
}
\arguments{
  \item{x}{
    An object of class \code{qreg.spike}.
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

\details{ The actual plotting will be handled by
  \code{\link{PlotMarginalInclusionProbabilities}},
  \code{\link{PlotLmSpikeCoefficients}}, or \code{\link{PlotModelSize}}.
  See the appropriate function for more options.}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{PlotMarginalInclusionProbabilities}}
  \code{\link{PlotLmSpikeCoefficients}}
  \code{\link{PlotModelSize}}
  \code{\link{qreg.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{predict.qreg.spike}}
}

\examples{
  n <- 50
  x <- rnorm(n)
  y <- rnorm(n, 4 * x)
  model <- qreg.spike(y ~ x,
                      quantile = .8,
                      niter = 1000,
                      expected.model.size = 100)
  plot(model)
  plot(model, "coef")
  plot(model, "coefficients")
  plot(model, "scaled.coefficients")
  plot(model, "scal")
  plot(model, "size")
  plot(model, "help")
}
\keyword{models}
\keyword{regression}
