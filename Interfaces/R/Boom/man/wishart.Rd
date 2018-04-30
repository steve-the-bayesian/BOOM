\name{wishart}
\title{Wishart Distribution}
\alias{dWishart}
\alias{rWishart}

\description{Density and random generation for the Wishart distribution.}

\usage{
dWishart(W, Sigma, nu, logscale = FALSE)
rWishart(nu, scale.matrix, inverse = FALSE)
}

\arguments{
  \item{W}{Argument (random variable) for the Wishart density.  A
    symmetric positive definite matrix.}

  \item{Sigma}{Scale or "variance" parameter of the Wishart
    distribution.  See the "details" section below.}

  \item{nu}{The "degrees of freedom" parameter of the Wishart
    distribution.  The distribution is only defined for \code{nu >=
      nrow(W) - 1}. }

  \item{logscale}{Logical.  If \code{TRUE} then the density is returned
    on the log scale.  Otherwise the density is returned on the density
    scale.}

  \item{scale.matrix}{For the Wishart distribution the
    \code{scale.matrix} parameter means the same thing as the
    \code{Sigma} parameter in \code{dWishart}.  It is the variance
    parameter of the generating multivariate normal distribution.

    If simulating from the inverse Wishart, \code{scale.matrix} is the
    INVERSE of the "sum of squares" matrix portion of the multivariate
    normal sufficient statistics.
  }

  \item{inverse}{Logical.  If TRUE then simulate from the inverse
    Wishart distribution.  If FALSE then simulate from the Wishart
    distribution.}
}

\details{

  If \code{nu} is an integer then a \eqn{W(\Sigma, \nu)}{W(Sigma, nu)}
  random variable can be thought of as the sum of \code{nu} outer
  products: \eqn{y_iy_i^T}{outer(y)}, where \eqn{y_i}{y} is a zero-mean
  multivariate normal with variance matrix \code{Sigma}.

  The Wishart distribution is

  \deqn{ \frac{|W|^{\frac{\nu - p - 1}{2}} \exp(-tr(\Sigma^{-1}W) / 2)}{
      2^{\frac{\nu p}{2}}|\Sigma|^{\frac{\nu}{2}}\Gamma_p(\nu / 2)}%
  }{
    det(W)^((nu-p-1)/2) * exp(-trace(solve(Sigma, W))) /
    (2^(nu * p / 2) * det(Sigma)^(nu / 2) * Gamma(nu/2, p))
  }
  where \code{p == nrow(W)} and \eqn{\Gamma_p(\nu)}{Gamma(nu, p)} is the
  multivariate gamma function (see \code{\link{lmgamma}}).
}

\value{
  \code{dWishart} returns the density of the Wishart distribution.  It
  is not vectorized, so only one random variable (matrix) can be
  evaluated at a time.

  \code{rWishart} returns one or more draws from the Wishart or inverse
  Wishart distributions.  If \code{n > 0} the result is a 3-way array.
  Unlike the \code{\link[stats]{rWishart}} function from the stats
  package, the first index corresponds to draws.  This is in keeping
  with the convention of other models from the Boom package.

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
