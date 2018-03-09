% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{month.distance}

\alias{MonthDistance}

\Rdversion{1.0}

\title{Elapsed time in months}

\description{The (integer) number of months between dates.}

\usage{
   MonthDistance(dates, origin)
}

\arguments{

  \item{dates}{A vector of class \code{\link{Date}} to be measured.}

  \item{origin}{A scalar of class \code{\link{Date}}.}

 }

 \value{ Returns a numeric vector giving the integer number of months
   that have elapsed between \code{origin} and each element in
   \code{dates}.  The daily component of each date is ignored, so two
   dates that are in the same month will have the same measured
   distance.  Distances are signed, so months that occur before
   \code{origin} will have negative values.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
  dates <- as.Date(c("2008-04-17",
                     "2008-05-01",
                     "2008-05-31",
                     "2008-06-01"))
  origin <- as.Date("2008-05-15")
  MonthDistance(dates, origin) ==  c(-1, 0, 0, 1)
}

\keyword{chron}
