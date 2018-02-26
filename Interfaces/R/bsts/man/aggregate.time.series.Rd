% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{aggregate.time.series}

\alias{AggregateTimeSeries}

\title{Aggregate a fine time series to a coarse summary}
\Rdversion{1.0}

\description{ Aggregate measurements from a fine scaled time series into
  a coarse time series.  This is similar to functions from the
  \code{xts} package, but it can handle aggregation from weeks to
  months.  }

\usage{
   AggregateTimeSeries(fine.series,
                       contains.end,
                       membership.fraction,
                       trim.left = any(membership.fraction < 1),
                       trim.right = NULL,
                       byrow = TRUE)
}

\arguments{

  \item{fine.series}{A numeric vector or matrix giving the fine scale
    time series to be aggregated.}

  \item{contains.end}{A logical vector corresponding to
    \code{fine.series} indicating whether each fine time interval
    contains the end of a coarse time interval.}

  \item{membership.fraction}{A numeric vector corresponding to
    \code{fine.series}, giving the fraction of each time interval's
    observation attributable to the coarse interval containing the fine
    interval's first day.  This will usually be a vector of 1's, unless
    \code{fine.series} is weekly.}

  \item{trim.left}{Logical indicating whether the first observation in
    the coarse aggregate should be removed.}

  \item{trim.right}{Logical indicating whether the final observation in
    the coarse aggregate should be removed.}

  \item{byrow}{Logical.  If \code{fine.series} is a matrix, this
    argument indicates whether rows (\code{TRUE}) or columns
    (\code{FALSE}) correspond to time points.}

}

\value{ A matrix (if \code{fine.series} is a matrix) or vector
  (otherwise) containing the aggregated values of \code{fine.series}.  }

\author{
  Steven L. Scott \email{stevescott@google.com}
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
  membership.fraction <- GetFractionOfDaysInInitialMonth(week.ending)
  which.month <- MatchWeekToMonth(week.ending, as.Date("2011-11-01"))
  contains.end <- WeekEndsMonth(week.ending)

  weekly.values <- rnorm(length(week.ending))
  monthly.values <- AggregateTimeSeries(weekly.values, contains.end, membership.fraction)

}

\keyword{chron}
