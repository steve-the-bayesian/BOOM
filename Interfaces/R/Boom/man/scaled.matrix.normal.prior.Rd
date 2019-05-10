% Copyright 2019 Steven L. Scott.  All rights reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{scaled.matrix.normal.prior}
\title{Scaled Matrix-Normal Prior}
\alias{ScaledMatrixNormalPrior}

\description{ A matrix-normal prior distribution, intended as the
  conjugate prior for the regression coefficients in a multivariate
  linear regression.}


\usage{
ScaledMatrixNormalPrior(mean, nu)
}

\arguments{

  \item{mean}{A matrix giving the mean of the distributions}

  \item{nu}{A scale factor affecting the variance.}
  
}

\details{

  The matrix normal distribution is a 3-parameter distribution MN(mu,
  Omega, V), where mu is the mean.  A deviate from the distribution is a
  matrix B, where Cov(B[i, j], B[k, m]) = Omega[i, k] * Sigma[j, m].  If
  b = Vec(B) is the vector obtained by stacking columns of B, then b is
  multivariate normal with mean Vec(mu) and covariance matrix
  \deqn{\Sigma \otimes Omega}{Sigma \%x\% Omega} (the kronecker
  product).

  This prior distribution assumes the underlying C++ code knows where to
  find the predictor (X) matrix in the regression, and the residual
  variance matrix Sigma.  The assumed prior distribution is
  B ~ MN(mu, X'X / nu, Sigma).

  Like most other priors in Boom, this function merely encodes
  information expected by the underlying C++ code, ensuring correct
  names and formatting.

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
