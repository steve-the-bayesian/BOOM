\name{check.data}
\title{Checking data formats}
\alias{check.data}
\alias{check.scalar.probability}
\alias{check.positive.scalar}
\alias{check.nonnegative.scalar}
\alias{check.probability.distribution}
\alias{check.scalar.integer}

\description{Checks that data matches a concept}

\usage{
check.scalar.probability(x)
check.positive.scalar(x)
check.nonnegative.scalar(x)
check.probability.distribution(x)
check.scalar.integer(x)
}

\arguments{
  \item{x}{An object to be checked.}
}

\details{ If the object does not match the concept being checked,
  \code{\link{stop}} is called.  Otherwise \code{TRUE} is returned.  }

\author{
  Steven L. Scott \email{stevescott@google.com}
}
