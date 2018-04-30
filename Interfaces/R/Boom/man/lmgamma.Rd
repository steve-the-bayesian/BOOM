\name{lmgamma}
\title{Log Multivariate Gamma Function}
\alias{lmgamma}

\description{Returns the log of the multivariate gamma function.}

\usage{
lmgamma(y, dimension)
}

\arguments{
  \item{y}{The function argument, which must be a positive scalar.}
  \item{dimension}{The dimension of the multivariate gamma function,
    which must be an integer >= 1.}
}

\details{
  The multivariate gamma function is

  \deqn{\Gamma_p(y) = \pi^{p(p-1)/4} \prod_{j =1}^p \Gamma(y + (1-j)/2).}{%
    Gamma(y, p) = pi^(p * (p-1)/4) prod(gamma(y + (1-(1:p))/2)).}

  The multivariate gamma function shows up as part of the normalizing
  constant for the Wishart and inverse Wishart distributions.
}

\value{ Returns the log of the multivariate gamma function.  Note that
  this function is not vectorized.  Both \code{y} and \code{dimension}
  must be scalars, and the return value is a scalar.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
