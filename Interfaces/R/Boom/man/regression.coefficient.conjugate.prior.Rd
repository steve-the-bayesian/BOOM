\name{regression.coefficient.conjugate.prior}
\title{Regression Coefficient Conjugate Prior}
\alias{RegressionCoefficientConjugatePrior}

\description{A conjugate prior for regression coefficients, conditional
  on residual variance, and sample size.}

\usage{
  RegressionCoefficientConjugatePrior(
    mean,
    sample.size,
    additional.prior.precision = numeric(0),
    diagonal.weight = 0)
}

\arguments{

  \item{mean}{The mean of the prior distribution, denoted 'b' below.
  See Details.}

  \item{sample.size}{ The value denoted \eqn{\kappa}{'kappa'} below.
    This can be interpreted as a number of observations worth of weight
    to be assigned to \code{mean} in the posterior distribution. }

  \item{additional.prior.precision}{ A vector of non-negative numbers
    representing the diagonal matrix \eqn{\Lambda^{-1}}{Lambda^{-1}}
    below.  Positive values for \code{additional.prior.precision} will
    ensure the distribution is proper even if the regression model has
    no data.  If all columns of the design matrix have positive variance
    then \code{additional.prior.precision} can safely be set to zero.  A
    zero-length numeric vector is a slightly more efficient equivalent
    to a vector of all zeros.}

  \item{diagonal.weight}{The weight given to the diagonal when XTX is
    averaged with its diagonal.  The purpose of \code{diagonal.weight}
    is to keep the prior distribution proper even if X is less than full
    rank.  If the design matrix is full rank then \code{diagonal.weight}
    can be set to zero.}

}

\details{

  A conditional prior for the coefficients (beta) in a linear regression
  model.  The prior is conditional on the residual variance
  \eqn{\sigma^2}{sigma^2}, the sample size n, and the design matrix X.
  The prior is

  \deqn{\beta | \sigma^2, X \sim %
    N(b, \sigma^2 (\Lambda^{-1} + V}{%
   beta | sigsq, X ~ N(b, sigsq * (Lambda^{-1} + V))}

 where
 \deqn{V^{-1} = \frac{\kappa}{n} ((1 - w) X^TX + w diag(X^TX)) .}{ %
   V^{-1} = ((1 - w) * XTX + w * Diag(XTX)) * kappa / n.}

   The prior distribution also depends on the cross product matrix
   XTX and the sample size n, which are not arguments to this
   function.  It is expected that the underlying C++ code will get
   those quantities elsewhere (presumably from the regression
   modeled by this prior).
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
