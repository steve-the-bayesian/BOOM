\name{check.process.specification}
\alias{CheckProcessSpecification}
\title{Check the validity of a process specification}

\description{Checks the valididity of process.specification for use in a
  \code{\link{MarkovModulatedPoissonProcess}}.}

\usage{
 CheckProcessSpecification(process.specification)
}

\arguments{ \item{process.specification}{A list of elements inheriting
  from \code{\link{PoissonProcessComponent}}}.  The elements must have
  non-empty names. }

\value{
  Returns invisible \code{NULL} if \code{process.specification} is
  valid.  Calls \code{\link{stop}} with an error message if it is not.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
