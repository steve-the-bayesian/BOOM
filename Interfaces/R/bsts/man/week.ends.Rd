% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{week.ends}

\alias{WeekEndsMonth}
\alias{WeekEndsQuarter}

\Rdversion{1.0}

\title{
  Check to see if a week contains the end of a month or quarter
}

\description{
  Returns a logical vector indicating whether the given week contains
  the end of a month or quarter.
}

\usage{
  WeekEndsMonth(week.ending)
  WeekEndsQuarter(week.ending)
}

\arguments{

  \item{week.ending}{A vector of class \code{\link{Date}}.  Each entry contains the
    date of the last day in a week.}

}

\value{A logical vector indicating whether the given week contains the
  end of a month or a quarter.}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{bsts.mixed}}.
}

\examples{
  week.ending <- as.Date(c("2011-10-01",
                           "2011-10-08",
                           "2011-12-03",
                           "2011-12-31"))
  WeekEndsMonth(week.ending) == c(TRUE, FALSE, TRUE, TRUE)
  WeekEndsQuarter(week.ending) == c(TRUE, FALSE, FALSE, TRUE)
}

\keyword{chron}
