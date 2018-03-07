\name{new.home.sales}
\docType{data}
\alias{new.home.sales}
\title{New home sales and Google trends}

\description{ The first column, HSN1FNSA is a time series of new home
  sales in the US, obtained from the FRED online data base.  The series
  has been manually deseasonalized.  The remaining columns contain
  search terms from Google trends (obtained from
  http://trends.google.com/correlate).  These show the relative
  popularity of each search term among all serach terms typed into
  Google.  All series in this data set have been standardized by
  subtracting off their mean and dividing by their standard deviation.}

\usage{data(new.home.sales)}
\format{zoo time series}
\source{FRED and trends.google.com}
\keyword{datasets}
