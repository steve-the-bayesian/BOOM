% Copyright 2010 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{SummarizeSpikeSlabCoefficients}
\Rdversion{1.1}
\alias{SummarizeSpikeSlabCoefficients}
\title{
  Numerical summaries of coefficients from a spike and slab regression.
}
\description{
  Produces a summary of the marginal distribution of model coefficients
  from a spike and slab regression.
}

\usage{
 SummarizeSpikeSlabCoefficients(beta, burn = 0, order = TRUE)
}
\arguments{

  \item{beta}{ A matrix containing MCMC draws of regression
    coefficients.  Each row is an MCMC draw.  Each column is a
    coefficient.
  }

  \item{burn}{
    The number of MCMC iterations in the ojbect to be discarded as
    burn-in.
  }

  \item{order}{ Logical.  If \code{TRUE} then the coefficients are
    presented in order of their posterior inclusion probabilities.
    Otherwise the order of the coefficients is the same as in
    \code{object}.  }
}

\value{

 A five-column matrix with rows representing model coefficients.  The
 first two columns are the posterior mean and standard deviation of each
 coefficient, including the point mass at zero.  The next two columns
 are the posterior mean and standard deviations conditional on the
 coefficient being nonzero.  The last column is the probability of a
 nonzero coefficient.

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
  plot(model)
  plot.ts(model$beta)
  hist(model$sigma)  ## should be near 8
  summary(model)
  SummarizeSpikeSlabCoefficients(model$beta, burn = 100)
}
\keyword{ models, regression}
