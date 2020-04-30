% Copyright 2010 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{plot.lm.spike}
\Rdversion{1.1}
\alias{plot.lm.spike}
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
  \method{plot}{lm.spike}(x,
     y = c("inclusion", "coefficients", "scaled.coefficients",
              "residuals", "fit", "size", "help"),
     burn = SuggestBurnLogLikelihood(x$log.likelihood),
     ...)
}
\arguments{
  \item{x}{
    An object of class \code{lm.spike}.
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
  \code{\link{PlotLmSpikeCoefficients}},
  \code{\link{PlotLmSpikeResiduals}}, or \code{\link{PlotModelSize}}.
  See the appropriate function for more options.}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{PlotMarginalInclusionProbabilities}}
  \code{\link{PlotLmSpikeCoefficients}}
  \code{\link{PlotLmSpikeResiduals}}
  \code{\link{PlotModelSize}}
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{summary.lm.spike}}
  \code{\link{predict.lm.spike}}
}

\examples{
simulate.lm.spike <- function(n = 100, p = 10, ngood = 3, niter=1000, sigma = 8){
  x <- cbind(matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  y <- rnorm(n, beta[1] + x \%*\% beta[-1], sigma)
  draws <- lm.spike(y ~ x, niter=niter)
  return(invisible(draws))
}
model <- simulate.lm.spike(n = 1000, p = 50, sigma = .3)
plot(model, inclusion.threshold = .01)

plot(model, "size")
}
\keyword{models}
\keyword{regression}
