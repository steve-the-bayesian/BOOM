\name{iclaims}
\docType{data}
\alias{initial.claims}
\alias{iclaims}
\title{Initial Claims Data}

\description{A weekly time series of US initial claims for unemployment.
  The first column contains the initial claims numbers from FRED.  The
  others contain a measure of the relative popularity of various search
  queries identified by Google Correlate.}

\usage{
data(iclaims)
}
\format{zoo time series}
\source{
  FRED.              http://research.stlouisfed.org/fred2/series/ICNSA,\cr
  Google correlate.  http://www.google.com/trends/correlate}
\examples{
data(iclaims)
plot(initial.claims)
}

\seealso{
  \code{\link{bsts}}
}

\keyword{datasets}
