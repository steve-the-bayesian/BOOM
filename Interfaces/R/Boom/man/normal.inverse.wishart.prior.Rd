% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{normal.inverse.wishart.prior}
\title{Normal inverse Wishart prior}
\alias{NormalInverseWishartPrior}

\description{The NormalInverseWishartPrior is the conjugate prior for the
  mean and variance of the multivariate normal distribution.  The model says
  that
  \deqn{\Sigma^{-1} \sim Wishart(\nu, S)  \mu|\sigma \sim N(\mu_0, \Sigma/\kappa)}{%
    \Sigma^{-1} ~ Wishart(\nu, S)   \mu | \Sigma ~ N(\mu0, \Sigma/\kappa)
  }

  The \eqn{Wishart(S, \nu)} distribution is parameterized by \code{S},
  the \emph{inverse} of the sum of squares matrix, and the scalar
  degrees of freedom parameter \code{nu}.

  The distribution is improper if \eqn{\nu < dim(S)}.
}


\usage{
NormalInverseWishartPrior(mean.guess,
                          mean.guess.weight = .01,
                          variance.guess,
                          variance.guess.weight = nrow(variance.guess) + 1)
}

\arguments{ \item{mean.guess}{The mean of the prior distribution.  This is
  \eqn{\mu_0}{\mu0} in the description above.}

  \item{mean.guess.weight}{The number of observations worth of weight
    assigned to \code{mean.guess}.  This is \eqn{\kappa}{\kappa} in the
    description above.}

  \item{variance.guess}{A prior estimate at the value of \eqn{\Sigma}.
    This is \eqn{S^{-1}/\nu} in the notation above.}

  \item{variance.guess.weight}{The number of observations worth of weight
    assigned to \code{variance.guess}.  This is \eqn{df}{df}.}
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
