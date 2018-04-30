% Copyright 2010 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{summary.logit.spike}
\Rdversion{1.1}
\alias{summary.logit.spike}
\alias{summary.probit.spike}
\title{
  Numerical summaries of the results from a spike and slab logistic regression.
}
\description{
  Produces a summary of the marginal distribution of model coefficients
  from a spike and slab logistic regression.
}

\usage{
\method{summary}{logit.spike}(object,
        burn = 0,
        order = TRUE,
        cutpoint.scale = c("probability", "logit"),
        cutpoint.basis = c("sample.size", "equal.range"),
        number.of.buckets = 10,
        coefficients = TRUE,
        ...)

\method{summary}{probit.spike}(object,
        burn = 0,
        order = TRUE,
        cutpoint.scale = c("probability", "probit"),
        cutpoint.basis = c("sample.size", "equal.range"),
        number.of.buckets = 10,
        coefficients = TRUE,
        ...)
}
\arguments{
  \item{object}{
    An object of class \code{logit.spike} or \code{probit.spike}.
  }

  \item{burn}{
    The number of MCMC iterations in the ojbect to be discarded as
    burn-in.
  }

  \item{order}{ Logical.  If \code{TRUE} then the coefficients are
    presented in order of their posterior inclusion probabilities.
    Otherwise the order of the coefficients is the same as in
    \code{object}.  }

  \item{cutpoint.scale}{The scale that should be used to determine the
    buckets for the comparison of predicted and actual probabilities.}

  \item{cutpoint.basis}{How should the buckets be determined in the
    comparison of predicted to actual probabilities?  If
    "sample.sample", then each bucket contains the same fraction of
    data.  If "equal.range" then the buckets are formed by parititioning
    the range of the predicted probabilities, and each bucket occupies
    the same amount of space on the real line.}

  \item{number.of.buckets}{The number of buckets to use in the
    comparison of predicted to actual probabilities.}

  \item{coefficients}{Logical value indicating whether the coefficient
    summary should be included in the output.  It can be useful to
    suppress the coefficients if there are many of them.}

  \item{\dots}{
    Unused.  Present for compatibility with generic summary().
  }

}

\value{
  Returns a list with the following elements
  \itemize{

    \item{coefficients: A five-column matrix summarizing the model
      coefficients, produced by
      \code{\link{SummarizeSpikeSlabCoefficients}}.}

    \item{null.log.likelihood: The log likelihood of the null binomial
      model evaluated at the MLE.}

    \item{mean.log.likelihood: The average value of log likelihood
      visited by the sampler.}

    \item{max.log.likelihood: The largest log likelihood value visited
      by the sampler.}

    \item{deviance.r2: The deviance R-square obtained by taking
      \code{(null.likelihood - mean.log.likelihood) /
        null.log.likelihood}}

    \item{deviance.r2.distribution: The value of the deviance R-square
      statistic at each point visited by the MCMC chain.  This is not
      printed by the print method.}

    \item{predicted.vs.actual}{A table obtained by paritioning the data
      into buckets, and comparing the aveage predicted probability with
      the empirical success rate in each bucket.}
  }
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{logit.spike}}
  \code{\link{probit.spike}}
  \code{\link{SpikeSlabPrior}}
}

\examples{
  n <- 100
  p <- 10
  ngood <- 3
  niter <- 1000

  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  prob <- plogis(x \%*\% beta)
  y <- runif(n) < prob
  x <- x[,-1]
  model <- logit.spike(y ~ x, niter=niter)
  summary(model)
}
\keyword{ models, regression}
