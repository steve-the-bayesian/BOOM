% Copyright 2012 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{compare.vector.distribution}
\title{Boxplots to compare distributions of vectors}

\alias{CompareVectorBoxplots}
\alias{compare.vector.boxplots}

\description{Uses boxplots to compare distributions of vectors.}

\usage{
CompareVectorBoxplots(draws, main = NULL, colors = NULL, burn = 0, \dots)
}

\arguments{

  \item{draws}{A list of MCMC draws.  Each list element is a matrix with
   rows corresponding to MCMC iterations and columns to variables.  The
   matrices can have different numbers of rows, but should have the same
   numbers of columns.}

 \item{main}{Main title of the plot.}

 \item{colors}{Colors to use for the boxplots.  The length must match
   the number entries in \code{draws}.}

 \item{burn}{The number of initial MCMC iterations to discard before
   making the plot.}

 \item{\dots}{Extra arguments passed to \code{\link{boxplot}}.}
}

\details{Creates side-by-side boxplots with the dimensions of each
  vector gropued together.}

\examples{

x <- matrix(rnorm(300, mean = 1:3, sd = .4), ncol = 3, byrow = TRUE)
y <- matrix(rnorm(600, mean = 3:1, sd = .2), ncol = 3, byrow = TRUE)
CompareVectorBoxplots(list(x = x, y = y), colors = c("red", "blue"))

}
