% Copyright 2010 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{print.summary.lm.spike}
\Rdversion{1.1}
\alias{print.summary.lm.spike}
\alias{print.summary.logit.spike}
\title{
  Print method for spikeslab objects.
}

\description{
  Print a spikeslab object.
}

\usage{
\method{print}{summary.lm.spike}(x, ...)
\method{print}{summary.logit.spike}(x, ...)
}
\arguments{
  \item{x}{
    An object of class \code{\link{summary.lm.spike}}.
  }

  \item{\dots}{
    Additional arguments passed to \code{\link{print.default}}.
  }

}

\value{
  This function is called for its side effect, which is to print the
  \code{spikeslab} object to the screen.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{summary.lm.spike}}
}

\examples{
  n <- 100
  p <- 10
  ngood <- 3
  niter <- 1000
  sigma <- 2

  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  y <- rnorm(n, x \%*\% beta, sigma)
  x <- x[,-1]
  model <- lm.spike(y ~ x, niter=niter)
  summary(model)
}
