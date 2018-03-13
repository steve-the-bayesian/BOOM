% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{get.fraction}

\alias{GetFractionOfDaysInInitialMonth}
\alias{GetFractionOfDaysInInitialQuarter}

\Rdversion{1.0}

\title{
  Compute membership fractions
}

\description{
  Returns the fraction of days in a week that occur in the ear
}

\usage{
   GetFractionOfDaysInInitialMonth(week.ending)
   GetFractionOfDaysInInitialQuarter(week.ending)
}

\arguments{
  \item{week.ending}{A vector of class \code{\link{Date}}.  Each entry contains the
    date of the last day in a week.}
}

\value{
  Returns a numeric vector of the same length as \code{week.ending}.
  Each entry gives the fraction of days in the week that occur in the
  coarse time interval (month or quarter) containing the start of the
  week (i.e the date 6 days before).
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{bsts.mixed}}.
}

\examples{
    dates <- as.Date(c("2003-03-31",
                       "2003-04-01",
                       "2003-04-02",
                       "2003-04-03",
                       "2003-04-04",
                       "2003-04-05",
                       "2003-04-06",
                       "2003-04-07"))
    fraction <- GetFractionOfDaysInInitialMonth(dates)
    fraction == c(1, 6/7, 5/7, 4/7, 3/7, 2/7, 1/7, 1)
}

\keyword{chron}
