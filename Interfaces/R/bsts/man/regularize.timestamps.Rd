\name{regularize.timestamps}

\alias{RegularizeTimestamps}
\alias{RegularizeTimestamps.default}
\alias{RegularizeTimestamps.numeric}
\alias{RegularizeTimestamps.Date}
\alias{RegularizeTimestamps.POSIXt}

\title{Produce a Regular Series of Time Stamps}
\Rdversion{1.0}

\description{
  Given an set of timestamps that might contain duplicates and gaps,
  produce a set of timestamps that has no duplicates and no gaps.
}

\usage{
  RegularizeTimestamps(timestamps)

  \method{RegularizeTimestamps}{default}(timestamps)

  \method{RegularizeTimestamps}{numeric}(timestamps)

  \method{RegularizeTimestamps}{Date}(timestamps)

  \method{RegularizeTimestamps}{POSIXt}(timestamps)
}

\arguments{

  \item{timestamps}{A set of (possibly irregular or non-unique)
    timestamps.  This could be a set of integers (like 1, 2, , 3...), a
    set of numeric like (1945, 1945.083, 1945.167, ...) indicating years
    and fractions of years, a \code{\link{Date}} object, or a
    \code{\link{POSIXt}} object.}  If the argument is \code{NULL} a
    \code{NULL} will be returned.}

\value{

  A set of regularly spaced timestamps of the same class as the argument
  (which might be \code{NULL}).

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
  first <- as.POSIXct("2015-04-19 08:00:04")
  monthly <- seq(from = first, length.out = 24, by = "month")
  skip.one <- monthly[-8]
  has.duplicates <- monthly
  has.duplicates[2] <- has.duplicates[3]

  reg1 <- RegularizeTimestamps(skip.one)
  all.equal(reg1, monthly) ## TRUE

  reg2 <- RegularizeTimestamps(has.duplicates)
  all.equal(reg2, monthly)  ## TRUE

}
\keyword{chron}
