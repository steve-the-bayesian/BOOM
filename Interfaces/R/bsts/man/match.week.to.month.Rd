% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{match.week.to.month}

\alias{MatchWeekToMonth}

\Rdversion{1.0}

\title{
  Find the month containing a week
}

\description{
  Returns the index of a month, in a sequence of months, that contains a
  given week.
}

\usage{
   MatchWeekToMonth(week.ending, origin.month)
}

\arguments{

  \item{week.ending}{A vector of class \code{\link{Date}}.  Each entry contains the
    date of the last day in a week.}

  \item{origin.month}{A \code{\link{Date}}, giving any day in the month
    to use as the origin of the sequence (month 1).}

}

\value{ The index of the month matching the month containing the first
  day in \code{week.ending}.  The origin is month 1.  It is the caller's
  responsibility to ensure that these indices correspond to legal values
  in a particular vector of months.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
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
