\name{MATCH.NumericTimestamps}
\title{Match Numeric Timestamps}

\alias{MATCH.NumericTimestamps}

\description{S3 generic method for MATCH function supplied in the zoo package.}

\usage{
  \method{MATCH}{NumericTimestamps}(x, table, nomatch = NA, \dots)
}

\arguments{
  \item{x}{A numeric set of timestamps.}

  \item{table}{A set of regular numeric timestamps to match against. }

  \item{nomatch}{The value to be returned in the case when no match is found.
    Note that it is coerced to integer.}

  \item{\dots}{Additional arguments passed to \code{\link{match}}. }
}

\details{
  Numeric timestamps match if they agree to 8 significant digits.
}

\value{
  Returns the index of the entry in \code{table} matched by each
  argument in \code{x}.  If an entry has no match then \code{nomatch} is
  returned at that position.
}

\seealso{
  \code{\link[zoo]{MATCH}}
}
