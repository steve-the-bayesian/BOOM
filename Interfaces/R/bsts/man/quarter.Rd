% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{quarter}

\alias{Quarter}

\title{Find the quarter in which a date occurs}
\Rdversion{1.0}
\description{
  Returns the quarter and year in which a date occurs.
}

\usage{
   Quarter(date)
}

\arguments{

  \item{date}{A vector convertible to \code{\link{POSIXlt}}.  A
  \code{\link{Date}} or \code{character} is fine.}

}

\value{ A numeric vector identifying the quarter that each element of
  \code{date} corresponds to, expressed as a number of years since 1900.
  Thus Q1-2000 is 100.00, and Q3-2007 is 107.50.  }

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\examples{
  Quarter(c("2008-02-29", "2008-04-29"))
  # [1] 108.00 108.25
}
\keyword{chron}
