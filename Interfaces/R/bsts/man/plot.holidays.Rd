\name{plot.holidays}
\title{Plot bsts holidays}

\alias{PlotHolidays}

\description{ Makes a series of side-by-side boxplots for all the
   holiday state components in model.}


\usage{
PlotHolidays(model, ylim = NULL, same.scale = TRUE, ...)
}


\arguments{
  \item{model}{A model fit by \code{\link{bsts}} containing one or more holiday
    state components.}

  \item{ylim}{Limits on the vertical axis of the plots.  If \code{ylim}
    is specified, all plots will have the same vertical axis.}


  \item{same.scale}{If ylim is \code{NULL}, this flag determines whether
    all plots share the same scale for there vertical axis
    (\code{same.scale == TRUE}), or each plot is independently scaled
    (\code{same.scale == FALSE}).}

  \item{\dots}{Extra arguments passed to \code{\link{boxplot}}.}
}


\value{
  Returns \code{invisible{NULL}}.
}

\examples{
## TODO(stevescott):  add examples
}

\seealso{
  \code{\link{bsts}}
  \code{\link{AddNamedHolidays}}
  \code{\link{AddFixedDateHoliday}}
  \code{\link{AddNthWeekdayInMonthHoliday}}
}
