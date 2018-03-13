\name{format.timestamps}

\alias{NoDuplicates}
\alias{NoGaps}
\alias{IsRegular}
\alias{HasDuplicateTimestamps}

\title{Checking for Regularity}
\Rdversion{1.0}

\description{

  Tools for checking if a series of timestamps is 'regular' meaning that
  it has no duplicates, and no gaps.  Checking for regularity can be
  tricky.  For example, if you have monthly observations with
  \code{\link{Date}} or \code{\link{POSIXt}} timestamps then gaps
  between timestamps can be 28, 29, 30, or 31 days, but the series is
  still "regular".

}

\usage{
  NoDuplicates(timestamps)
  NoGaps(timestamps)
  IsRegular(timestamps)

  HasDuplicateTimestamps(bsts.object)
}

\arguments{

  \item{timestamps}{A set of (possibly irregular or non-unique)
    timestamps.  This could be a set of integers (like 1, 2, , 3...), a
    set of numeric like (1945, 1945.083, 1945.167, ...) indicating years
    and fractions of years, a \code{\link{Date}} object, or a
    \code{\link{POSIXt}} object.}

  \item{bsts.object}{A bsts model object.}
}

\value{
  All four functions return scalar logical values.  \code{NoDuplicates}
  returns \code{TRUE} if all elements of \code{timestamps} are unique.

  \code{NoGaps} examines the smallest nonzero gap between time points.
  As long as no gaps between time points are more than twice as wide as
  the smallest gap, it returns \code{TRUE}, indicating that there are no
  missing timestamps.  Otherwise it returns \code{FALSE}.

  \code{IsRegular} returns \code{TRUE} if \code{NoDuplicates} and
  \code{NoGaps} both return \code{TRUE}.

  \code{HasDuplicateTimestamps} returns \code{FALSE} if the data used to
  fit bsts.model either has NULL timestamps, or if the timestamps
  contain no duplicate values.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
  first <- as.POSIXct("2015-04-19 08:00:04")
  monthly <- seq(from = first, length.out = 24, by = "month")
  IsRegular(monthly) ## TRUE

  skip.one <- monthly[-8]
  IsRegular(skip.one) ## FALSE

  has.duplicates <- monthly
  has.duplicates[1] <- has.duplicates[2]
  IsRegular(has.duplicates) ## FALSE
}
\keyword{chron}
