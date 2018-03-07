\name{plot.logit.spike.residuals}
\Rdversion{1.1}
\alias{PlotLogitSpikeResiduals}
\alias{PlotProbitSpikeResiduals}
\title{
  Residual plot for \code{\link{logit.spike}} objects.
}
\description{
  Plots the "deviance residuals" from a logit.spike model.

}

\usage{
  PlotLogitSpikeResiduals(model, ...)
  PlotProbitSpikeResiduals(model, ...)
}

\arguments{
  \item{model}{A model object inheriting from \code{\link{logit.spike}}
  or \code{\link{probit.spike}}.}

  \item{\dots}{
    Additional arguments to be passed to \code{\link{plot}.}
  }
}

\details{
 The "deviance residuals" are defined as the signed square root
 each observation's contribution to log likelihood.  The sign of
 the residual is positive if half or more of the trials associated
 with an observation are successes.  The sign is negative
 otherwise.

  The "contribution to log likelihood" is taken to be the posterior mean
  of an observations log likelihood contribution, averaged over the life
  of the MCMC chain.

  The deviance residual is plotted against the fitted value, again
  averaged over the life of the MCMC chain.

  The plot also shows the .95 and .99 bounds from the square root
  of a chi-square(1) random variable.  As a rough approximation,
  about 5\% and 1\% of the data should lie outside these bounds.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{logit.spike}}
  \code{\link{plot.logit.spike}}
}

\examples{
simulate.logit.spike <- function(n = 100, p = 10, ngood = 3,
                              niter=1000){
  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  prob <- plogis(x \%*\% beta)
  y <- runif(n) < prob
  x <- x[,-1]
  draws <- logit.spike(y ~ x, niter=niter)
  plot.ts(draws$beta)
  return(invisible(draws))
}
model <- simulate.logit.spike()
plot(model, "fit")
plot(model, "fit", scale = "probability", number.of.buckets = 15)
}
\keyword{models}
\keyword{regression}
