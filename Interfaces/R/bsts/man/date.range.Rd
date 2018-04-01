\name{date.range}

\alias{DateRange}

\Rdversion{1.0}

\title{
  Date Range
}

\description{
   Returns the first and last dates of the influence window for the given
   holiday, among the given timestamps.
}

\usage{
  DateRange(holiday, timestamps)
}

\arguments{

  \item{holiday}{An object of class \code{\link{Holiday}}.}

  \item{timestamps}{A vector of timestamps of class \code{\link{Date}}
    or class \code{\link{POSIXt}}.  This function assumes daily
    data.  Use with care in other settings.}
  
}

\value{ Returns a two-column data frame giving the first and last dates
  of the influence window for the holiday in the period covered by
  \code{timestamps}.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{

holiday <- NamedHoliday("MemorialDay", days.before = 2, days.after = 2)
timestamps <- seq.Date(from = as.Date("2001-01-01"), by = "day",
   length.out = 365 * 10)
influence <- DateRange(holiday, timestamps)

}

\keyword{chron}
