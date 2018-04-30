\name{inverse-wishart}
\title{Inverse Wishart Distribution}
\alias{dInverseWishart}
\alias{InverseWishartPrior}

\description{Density for the inverse Wishart distribution.}

\usage{
dInverseWishart(Sigma, sum.of.squares, nu, logscale = FALSE,
                log.det.sumsq = log(det(sum.of.squares)))

InverseWishartPrior(variance.guess, variance.guess.weight)
}

\arguments{
  \item{Sigma}{Argument (random variable) for the inverse Wishart
    distribution.  A positive definite matrix.}

  \item{nu}{The "degrees of freedom" parameter of the inverse Wishart
    distribution.  The distribution is only defined for \code{nu >=
      nrow(Sigma) - 1}. }

  \item{sum.of.squares}{A positive definite matrix.  Typically this is
    the sum of squares that is the sufficient statistic for the
    inverse Wishart distribution.}

  \item{logscale}{Logical.  If \code{TRUE} then the density is returned
    on the log scale.  Otherwise the density is returned on the density
    scale.}

  \item{log.det.sumsq}{The log determinant of \code{sum.of.squares}.  If
    this function is to be called many times then precomputing the log
    determinant can save considerable compute time. }

  \item{variance.guess}{A prior guess at the value of the variance
    matrix the prior is modeling.}

  \item{variance.guess.weight}{A positive scalar indicating the number
    of observations worth of weight to place on \code{variance.guess}.}
}

\details{
  The inverse Wishart distribution has density function
  \deqn{ \frac{|Sigma|^{-\frac{\nu + p + 1}{2}} \exp(-tr(\Sigma^{-1}S) / 2)}{
      2^{\frac{\nu p}{2}}|\Sigma|^{\frac{\nu}{2}}\Gamma_p(\nu / 2)}%
  }{
    det(Sigma)^(-(nu+p+1)/2) * exp(-trace(solve(Sigma, S))) /
    (2^(nu * p / 2) * det(Sigma)^(nu / 2) * Gamma(nu/2, p))
  }
}

\value{

  \code{dInverseWishart} returns the scalar density (or log density) at
  the specified value.  This function is not vectorized, so only one
  random variable (matrix) can be evaluated at a time.

  \code{InverseWishartPrior} returns a list that encodes the parameters
  of the distribution in a format expected by underlying C++ code.  }

\seealso{
  \code{\link{dWishart}},
  \code{\link{rWishart}},
  \code{\link{NormalInverseWishartPrior}}
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
