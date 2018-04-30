\name{descriptive-plots}
\title{Descriptive Plots}

\alias{YearPlot}
\alias{MonthPlot}
\alias{DayPlot}

\description{Plots for describing time series data.}

\usage{
  DayPlot(y, colors = NULL, ylab = NULL, ...)
  MonthPlot(y, seasonal.identifier = months, colors = NULL, ylab = NULL, ...)
  YearPlot(y, colors = NULL, ylab = NULL, ylim = NULL, legend = TRUE, ...)
}

\arguments{

  \item{y}{A time series to plot.  Must be of class \code{\link{ts}}, or
    \code{\link{zoo}}.  If a zoo object then the timestamps must be of
    type \code{\link{Date}}, \code{\link{yearmon}}, or \code{\link{POSIXt}}.
  }

  \item{seasonal.identifier}{
    A function that takes a vector of class \code{\link{POSIXt}}
    (date/time) and returns a character vector indicating the season to
    which each element belongs.  Each unique element returned by this
    function returns a "season" to be plotted.  See
    \code{\link{weekdays}}, \code{\link{months}},
    and \code{\link{quarters}} for examples of how this should work.  }

  \item{colors}{A vector of colors to use for the lines.}

  \item{legend}{Logical.  If \code{TRUE} then a legend is added to the plot.}
  
  \item{ylab}{Label for the vertical axis.}

  \item{ylim}{Limits for the vertical axis. (a 2-vector)}

  \item{\dots}{Extra arguments passed to \code{\link{plot}} or \code{\link{lines}}.}

}

\details{

  \code{DayPlot} and \code{MonthPlot} plot the time series one season at
  a time, on the same set of axes.  The intent is to use DayPlot for
  daily data and MonthPlot for monthly or quarterly data.

  \code{YearPlot} plots each year of the time series as a separate line
  on the same set of axes.

  Both sets of plots help visualize seasonal patterns.
  
}

\value{
  Returns \code{invisible{NULL}}.
}

\examples{
## Plot a 'ts' time series.
data(AirPassengers)
par(mfrow = c(1,2))
MonthPlot(AirPassengers)
YearPlot(AirPassengers)

## Plot a 'zoo' time series.
data(turkish)
par(mfrow = c(1,2))
YearPlot(turkish)
DayPlot(turkish)

}
  
\seealso{ \code{\link{monthplot}} is a base R function for plotting time
  series of type \code{\link{ts}}.  }
