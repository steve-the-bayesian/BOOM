\name{residuals.lm.spike}
\Rdversion{1.1}

\alias{residuals.lm.spike}

\title{
  Extract lm.spike Residuals
}

\description{
  Get residuals from an \code{\link{lm.spike}} object.
}

\usage{
\method{residuals}{lm.spike}(
    object,
    burn = SuggestBurnLogLikelihood(object$log.likelihood),
    mean.only = FALSE,
    \dots)
}

\arguments{
  \item{object}{
    An object of class \code{lm.spike}.
  }

  \item{burn}{
    The number of MCMC iterations in the object to be discarded as
    burn-in.
  }

  \item{mean.only}{ Logical.  If \code{TRUE} then the posterior mean of
    each residual is returned.  If \code{FALSE} then the full
    posterior distribution of residuals is returned.  }

  \item{\dots}{Unused, but present for compatibility with generic
    \code{residuals} function.}

}

\value{ The posterior distribution (or posterior mean) of residuals from
  the model object.  If \code{mean.only} is \code{TRUE} then the return
  value is the vector of residuals, otherwise the return value is a
  matrix, with rows corresponding to MCMC iterations, and columns to
  individual observations.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{summary.lm.spike}}
  \code{\link{plot.lm.spike}}
}

\examples{
  niter <- 1000
  n <- 100
  p <- 10
  ngood <- 3

  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- rep(0, p)
  good <- sample(1:p, ngood)
  beta[good] <- rnorm(ngood)
  sigma <- 1

  y <- rnorm(n, x \%*\% beta, sigma)
  model <- lm.spike(y ~ x - 1, niter=niter)
  plot(model)
  residuals(model)
  residuals(model, mean.only = TRUE)
}
\keyword{models, regression}
