% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{sufstat.Rd}
\title{Sufficient Statistics}
\alias{RegressionSuf}
\alias{GaussianSuf}

\description{Sufficient statistics for various models.}

\usage{
  RegressionSuf(X = NULL,
                y = NULL,
                xtx = crossprod(X),
                xty = crossprod(X, y),
                yty = sum(y^2),
                n = length(y),
                xbar = colMeans(X),
                ybar = mean(y))

  GaussianSuf(y)
}

\arguments{
  \item{X}{The predictor matrix for a regression problem.}

  \item{y}{The data, or the regression response variable.}

  \item{xtx}{The cross product of the design matrix. "X transpose X."}

  \item{xty}{The cross product of the design matrix with the response
    vector. "X transpose y."}

  \item{yty}{The sum of the squares of the response vector.
    "y transpose y."}

  \item{n}{The sample size.}

  \item{xbar}{A vector giving the average of each column in the
    predictor matrix.}

  \item{ybar}{The (scalar) mean of the response variable y.}
}

\value{ The returned value is a function containing the sufficient
  statistics for a regression model.  Arguments are checked to ensure
  they have legal values.  List names match the names expected by
  underlying C++ code.  }

\examples{

  X <- cbind(1, matrix(rnorm(3 * 100), ncol = 3))
  y <- rnorm(100)

  ## Sufficient statistics can be computed from raw data, if it is
  ## available.
  suf1 <- RegressionSuf(X, y)

  ## The individual components can also be computed elsewhere, and
  ## provided as arguments.  If n is very large, this can be a
  ## substantial coomputational savings.
  suf2 <- RegressionSuf(xtx = crossprod(X),
                        xty = crossprod(X, y),
                        yty = sum(y^2),
                        n = 100,
                        xbar = colMeans(X))

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
