% Copyright 2010 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{summary.lm.spike}
\Rdversion{1.1}
\alias{summary.lm.spike}
\title{
  Numerical summaries of the results from a spike and slab regression.
}
\description{
  Produces a summary of the marginal distribution of model coefficients
  from a spike and slab regression.
}

\usage{
\method{summary}{lm.spike}(object, burn = 0, order = TRUE, ...)
}
\arguments{
  \item{object}{
    An object of class \code{lm.spike}.
  }

  \item{burn}{
    The number of MCMC iterations in the ojbect to be discarded as
    burn-in.
  }

  \item{order}{ Logical.  If \code{TRUE} then the coefficients are
    presented in order of their posterior inclusion probabilities.
    Otherwise the order of the coefficients is the same as in
    \code{object}.  }

  \item{\dots}{
    Unused.  Present for compatibility with generic summary().
  }

}

\value{

  Returns a list with the following elements:

  \item{coefficients}{ A five-column matrix with rows representing model
    coefficients.  The first two columns are the posterior mean and
    standard deviation of each coefficient, including the point mass at
    zero.  The next two columns are the posterior mean and standard
    deviations conditional on the coefficient being nonzero.  The last
    column is the probability of a nonzero coefficient.}

  \item{residual.sd}{A summary of the posterior distribution of the
    residual standard deviation parameter.}

  \item{rsquare}{A summary of the posterior distribution of the R^2
    statistic:  1 - residual.sd^2 / var(y)}

}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{plot.lm.spike}}
  \code{\link{predict.lm.spike}}
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
}
