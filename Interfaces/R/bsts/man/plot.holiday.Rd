\name{plot.holiday}
\title{Plot Holiday Effects}
\alias{PlotHoliday}

\description{ Plot the estimated effect of the given holiday.}

\usage{
  PlotHoliday(holiday, model, show.raw.data = TRUE, ylim = NULL, \dots)
}

\arguments{
  \item{holiday}{An object of class \code{\link{Holiday}}.}

  \item{model}{A model fit by \code{\link{bsts}} containing either a
    \code{\link{RegressionHolidayStateModel}} or
    \code{\link{HierarchicalRegressionHolidayStateModel}} that includes
    \code{holiday}.  }

  \item{show.raw.data}{Logical indicating if the raw data corresponding
    to \code{holiday} should be superimposed on the plot.  The 'raw
    data' are the actual values of the target series, minus the value of
    the target series the day before the holiday began, which is a
    (somewhat poor) proxy for remaining state elements.  The raw data
    can appear artificially noisy if there are other strong state
    effects such as a day-of-week effect for holidays that don't always
    occur on the same day of the week.  }

  \item{ylim}{Limits on the vertical axis of the plots.}

  \item{\dots}{Extra arguments passed to \code{\link{boxplot}}.}
}


\value{
  Returns \code{invisible{NULL}}.
}

\examples{
  trend <- cumsum(rnorm(730, 0, .1))
  dates <- seq.Date(from = as.Date("2014-01-01"), length = length(trend),
    by = "day")
  y <- zoo(trend + rnorm(length(trend), 0, .2), dates)

  AddHolidayEffect <- function(y, dates, effect) {
    ## Adds a holiday effect to simulated data.
    ## Args:
    ##   y: A zoo time series, with Dates for indices.
    ##   dates: The dates of the holidays.
    ##   effect: A vector of holiday effects of odd length.  The central effect is
    ##     the main holiday, with a symmetric influence window on either side.
    ## Returns:
    ##   y, with the holiday effects added.
    time <- dates - (length(effect) - 1) / 2
    for (i in 1:length(effect)) {
      y[time] <- y[time] + effect[i]
      time <- time + 1
    }
    return(y)
  }

  ## Define some holidays.
  memorial.day <- NamedHoliday("MemorialDay")
  memorial.day.effect <- c(.3, 3, .5)
  memorial.day.dates <- as.Date(c("2014-05-26", "2015-05-25"))
  y <- AddHolidayEffect(y, memorial.day.dates, memorial.day.effect)
  
  presidents.day <- NamedHoliday("PresidentsDay")
  presidents.day.effect <- c(.5, 2, .25)
  presidents.day.dates <- as.Date(c("2014-02-17", "2015-02-16"))
  y <- AddHolidayEffect(y, presidents.day.dates, presidents.day.effect)
  
  labor.day <- NamedHoliday("LaborDay")
  labor.day.effect <- c(1, 2, 1)
  labor.day.dates <- as.Date(c("2014-09-01", "2015-09-07"))
  y <- AddHolidayEffect(y, labor.day.dates, labor.day.effect)
  
  ## The holidays can be in any order.
  holiday.list <- list(memorial.day, labor.day, presidents.day)
  number.of.holidays <- length(holiday.list)
  
  ## In a real example you'd want more than 100 MCMC iterations.
  niter <- 100
  ss <- AddLocalLevel(list(), y)
  ss <- AddRegressionHoliday(ss, y, holiday.list = holiday.list)
  model <- bsts(y, state.specification = ss, niter = niter)

  PlotHoliday(memorial.day, model)
}
  
\seealso{
  \code{\link{bsts}}
  \code{\link{AddRandomWalkHoliday}}
}
