% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{extend.time}

\alias{ExtendTime}

\Rdversion{1.0}

\title{
  Extends a vector of dates to a given length
}

\description{
  Pads a vector of dates to a specified length.
}

\usage{
   ExtendTime(dates, number.of.periods, dt = NULL)
}

\arguments{

  \item{dates}{An ordered vector of class \code{\link{Date}}.}

  \item{number.of.periods}{The desired length of the output.}

  \item{dt}{A character string describing the frequency of the dates in
    \code{dates}.  Possible values are "daily", "weekly", "monthly",
    "quarterly", "yearly", or "other".  An attempt to deduce \code{dt}
    will be made if it is missing.}

}

\value{
  If \code{number.of.periods} is longer than \code{length(dates)}, then
  \code{dates} will be padded to the desired length.  Extra dates are
  added at time intervals matching the average interval in
  \code{dates}.  Thus they may not be
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\seealso{
  \code{\link{bsts.mixed}}.
}

\examples{
  origin.month <- as.Date("2011-09-01")
  week.ending <- as.Date(c("2011-10-01",   ## 1
                           "2011-10-08",   ## 2
                           "2011-12-03",   ## 3
                           "2011-12-31"))  ## 4
  MatchWeekToMonth(week.ending, origin.month) == 1:4
}

\keyword{chron}
