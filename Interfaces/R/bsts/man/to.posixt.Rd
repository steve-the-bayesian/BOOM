\name{to.posixt}
\alias{YearMonToPOSIX}
\alias{DateToPOSIX}
\title{Convert to POSIXt}

\Rdversion{1.1}

\description{
 Convert an object of class Date to class POSIXct without getting bogged
 down in timezone calculation.

}

\details{

  Calling \code{\link{as.POSIXct}} on another date/time object
  (e.g. Date) applies a timezone correction to the object.  This can
  shift the time marker by a few hours, which can have the effect of
  shifting the day by one unit.  If the day was the first or last in a
  month or year, then the month or year will be off by one as well.
  
  Coercing the object to the character representation of a Date prevents this
  adjustment from being applied, and leaves the POSIXt return value with the
  intended day, month, and year.
}

\usage{
  DateToPOSIX(timestamps)
  YearMonToPOSIX(timestamps)
}

\arguments{

  \item{timestamps}{An object of class \code{\link{yearmon}} or
  \code{\link{Date}} to be converted to POSIXct.}
  
}


\author{
  Steven L. Scott  \email{steve.the.bayesian@gmail.com}
}


\keyword{models}
\keyword{regression}
