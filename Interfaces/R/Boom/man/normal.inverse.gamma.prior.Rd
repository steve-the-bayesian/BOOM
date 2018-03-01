% Copyright 2012 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{normal.inverse.gamma.prior}
\title{Normal inverse gamma prior}
\alias{NormalInverseGammaPrior}

\description{The NormalInverseGammaPrior is the conjugate prior for the
  mean and variance of the scalar normal distribution.  The model says
  that
  \deqn{\frac{1}{\sigma^2} \sim Gamma(df / 2, ss/2)  \mu|\sigma \sim
  N(\mu_0, \sigma^2/\kappa)}{%
    1/\sigma^2 ~ Gamma(df/2, ss/2)   \mu | \sigma ~ N(\mu0, \sigma^2/\kappa)
  }
}


\usage{
NormalInverseGammaPrior(mu.guess, mu.guess.weight = .01,
       sigma.guess, sigma.guess.weight = 1, ...)

}

\arguments{ \item{mu.guess}{The mean of the prior distribution.  This is
  \eqn{\mu_0}{\mu0} in the description above.}

  \item{mu.guess.weight}{The number of observations worth of weight
    assigned to \code{mu.guess}.  This is \eqn{\kappa}{\kappa} in the
    description above.}

  \item{sigma.guess}{A prior estimate at the value of \code{sigma}.
    This is \eqn{\sqrt{ss/df}}{\sqrt{ss/df}}.}

  \item{sigma.guess.weight}{The number of observations worth of weight
    assigned to \code{sigma.guess}.  This is \eqn{df}{df}.}

  \item{...}{blah}
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
