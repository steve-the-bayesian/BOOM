% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)


\name{last.day.in.month}

\alias{LastDayInMonth}

\title{Find the last day in a month}
\Rdversion{1.0}
\description{
  Finds the last day in the month containing a specefied date.
}

\usage{
   LastDayInMonth(dates)
}

\arguments{

  \item{dates}{A vector of class \code{\link{Date}}.}

}

\value{
  A vector of class \code{\link{Date}} where each entry is the last day
  in the month containing the corresponding entry in \code{dates}.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}


\examples{
 inputs <- as.Date(c("2007-01-01",
                     "2007-01-31",
                     "2008-02-01",
                     "2008-02-29",
                     "2008-03-14",
                     "2008-12-01",
                     "2008-12-31"))
 expected.outputs <- as.Date(c("2007-01-31",
                              "2007-01-31",
                              "2008-02-29",
                              "2008-02-29",
                              "2008-03-31",
                              "2008-12-31",
                              "2008-12-31"))
 LastDayInMonth(inputs) == expected.outputs
}

\keyword{chron}
