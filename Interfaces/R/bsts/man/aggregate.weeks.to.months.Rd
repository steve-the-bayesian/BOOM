% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{aggregate.weeks.to.months}

\alias{AggregateWeeksToMonths}

\title{Aggregate a weekly time series to monthly}
\Rdversion{1.0}

\description{Aggregate measurements from a weekly time series into
  a monthly time series.  }

\usage{
   AggregateWeeksToMonths(weekly.series,
                          membership.fraction = NULL,
                          trim.left = TRUE,
                          trim.right = NULL)

}

\arguments{

  \item{weekly.series}{A numeric vector or matrix of class
    \code{\link[zoo]{zoo}} giving the weekly time series to be aggregated.
    The index must be convertible to class \code{\link{Date}}.}

  \item{membership.fraction}{A optional numeric vector corresponding to
    \code{weekly.series}, giving the fraction of each week's observation
    attributable to the month containing the week's first day.  If
    missing, then weeks will be split across months in proportion to the
    number of days in each month.}

  \item{trim.left}{Logical indicating whether the first observation in
    the monthly aggregate should be removed.}

  \item{trim.right}{Logical indicating whether the final observation in
    the monthly aggregate should be removed.}

}

\value{ A zoo-matrix (if \code{weekly.series} is a matrix) or vector
  (otherwise) containing the aggregated values of \code{weekly.series}.
  }

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\seealso{
  \code{\link{AggregateTimeSeries}}
}

\examples{
  week.ending <- as.Date(c("2011-11-05",
                           "2011-11-12",
                           "2011-11-19",
                           "2011-11-26",
                           "2011-12-03",
                           "2011-12-10",
                           "2011-12-17",
                           "2011-12-24",
                           "2011-12-31"))

  weekly.values <- zoo(rnorm(length(week.ending)), week.ending)
  monthly.values <- AggregateWeeksToMonths(weekly.values)
}

\keyword{chron}
