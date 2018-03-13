\name{residuals.bsts}
\title{Residuals from a bsts Object}

\alias{residuals.bsts}

\description{Residuals (or posterior distribution of residuals) from a
  bsts object.}

\usage{
  \method{residuals}{bsts}(object,
    burn = SuggestBurn(.1, object),
    mean.only = FALSE,
    \dots)
}

\arguments{
  \item{object}{An object of class \code{\link{bsts}} created by the function
    of the same name.}

  \item{burn}{The number of MCMC iterations to discard as burn-in.}

  \item{mean.only}{Logical.  If \code{TRUE} then the mean residual for
    each time period is returned.  If \code{FALSE} then the full
    posterior distribution is returned.}

  \item{\dots}{Not used.  This argument is here to comply with the
    signature of the generic residuals function.}
}

\value{

  If \code{mean.only} is \code{TRUE} then this function returns a vector
  of residuals with the same "time stamp" as the original series.  If
  \code{mean.only} is \code{FALSE} then the posterior distribution of
  the residuals is returned instead, as a matrix of draws.  Each row of
  the matrix is an MCMC draw, and each column is a time point.  The
  colnames of the returned matrix will be the timestamps of the original
  series, as text.  }


\seealso{
  \code{\link{bsts}}, \code{\link{plot.bsts}}.
}
