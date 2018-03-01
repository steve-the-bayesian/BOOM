% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{estimate.time.scale}
\title{Intervals between dates}
\alias{EstimateTimeScale}

\description{Estimate the time scale used in time series data.}

\usage{EstimateTimeScale(dates)}

\arguments{
  \item{dates}{A sorted vector of class \code{\link{Date}}.}
}

\value{A character string.  Either "daily", "weekly", "yearly",
  "monthly", "quarterly", or "other".  The value is determined based on
  counting the number of days between successive observations in \code{dates}.}

\author{Steven L. Scott \email{stevescott@google.com}}

\examples{
weekly.data <- as.Date(c("2011-10-01",
                         "2011-10-08",
                         "2011-10-15",
                         "2011-10-22",
                         "2011-10-29",
                         "2011-11-05"))

EstimateTimeScale(weekly.data) # "weekly"

almost.weekly.data <- as.Date(c("2011-10-01",
                                "2011-10-08",
                                "2011-10-15",
                                "2011-10-22",
                                "2011-10-29",
                                "2011-11-06"))  # last day is one later

EstimateTimeScale(weekly.data) # "other"
}

\keyword{chron}


