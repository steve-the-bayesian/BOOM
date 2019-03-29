\name{gdp}
\docType{data}
\alias{gdp}
\title{Gross Domestic Product for 57 Countries}
\description{Annual gross domestic product for 57 countries, as produced
  by the OECD.  

  Fields:
  \itemize{
    \item{LOCATION: Three letter country code.}
    \item{MEASURE: MLN_USD signifies a total GDP number in millions of
      US dollars.  USD_CAP is per capita GDP in US dollars.}
    \item{TIME: The year of the measurement.}
    \item{Value: The measured value.}
    \item{Flag.Codes: P for provisional data, B for a break in the
  series, and E for an estimated value.}
    }
}
\usage{
data(gdp)
}
\format{data frame}
\source{OECD website: See
  https://data.oecd.org/gdp/gross-domestic-product-gdp.htm}

\keyword{datasets}
