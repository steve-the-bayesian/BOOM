\name{wide.to.long}
\alias{WideToLong}
\alias{LongToWide}

\Rdversion{1.1}

\title{
  Convert Between Wide and Long Format
}

\description{

  Convert a multivariate time series between wide and long formats.  In
  "wide" format there is one row per time point, with series organzied
  by columns.  In "long" format there is one row per observation, with
  variables indicating the series and time point to which an observation
  belongs.
  
}

\usage{
  WideToLong(response, na.rm = TRUE)
  LongToWide(response, series.id, timestamps)
}

\arguments{

  \item{response}{For \code{WideToLong} this is a matrix, with rows
    representing time and columns representing variables.  This can be a
    \code{\link{zoo}} matrix with timestamps as an index.

    For \code{LongToWide}, \code{response} is a vector.}

  \item{na.rm}{If TRUE then missing values will be omitted from the
    returned data frame (their absence denoting missingness).
    Otherwise, missing values will be included as NA's.}

  \item{series.id}{A factor (or variable coercible to factor) of the
    same length as \code{response}, indicating the series to which each
    observation belongs. }

  \item{timestamps}{A variable of the same length as \code{response},
    indicating the time period to which each observation belongs.}  
}

\value{
  \code{LongToWide} returns a zoo matrix with the time series in wide format.
  \code{WideToLong} returns a 3-column data frame with columns "time", "series", and "values".
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
data(gdp)
gdp.wide <- LongToWide(gdp$GDP, gdp$Country, gdp$Time)
gdp.long <- WideToLong(gdp.wide)
}

