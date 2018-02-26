\name{HarveyCumulator}
\alias{HarveyCumulator}
\title{HarveyCumulator}
\Rdversion{1.1}

\description{ Given a state space model on a fine scale, the Harvey
  cumulator aggregates the model to a coarser scale (e.g. from days to
  weeks, or weeks to months).

}

\usage{
HarveyCumulator(fine.series,
                contains.end,
                membership.fraction)
}

\arguments{

  \item{fine.series}{The fine-scale time series to be aggregated.}

  \item{contains.end}{A logical vector, with length matching
    \code{fine.series} indicating whether each fine scale time interval
    contains the end of a coarse time interval.  For example, months
    don't contain a fixed number of weeks, so when cumulating a weekly
    time series into a monthly series, you need to know which weeks
    contain the end of a month.}

  \item{membership.fraction}{The fraction of each fine-scale time
    observation belonging to the coarse scale time observation at the
    beginning of the time interval.  For example, if week i started in
    March and ended in April, \code{membership.fraction[i]} is the
    fraction of fine.series[i] that should be attributed to March.  This
    should be 1 for most observations.}

}

\value{ Returns a vector containing the course scale partial aggregates
  of \code{fine.series}.  }

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\seealso{
  \code{\link{bsts.mixed}},
}

\examples{

data(goog)
days <- factor(weekdays(index(goog)),
               levels = c("Monday", "Tuesday", "Wednesday",
                          "Thursday", "Friday"),
               ordered = TRUE)

## Because of holidays, etc the days do not always go in sequence.
## (Sorry, Rebecca Black! https://www.youtube.com/watch?v=kfVsfOSbJY0)
## diff.days[i] is the number of days between days[i-1] and days[i].
## We know that days[i] is the end of a week if diff.days[i] < 0.
diff.days <- tail(as.numeric(days), -1) - head(as.numeric(days), -1)
contains.end <- c(FALSE, diff.days < 0)

goog.weekly <- HarveyCumulator(goog, contains.end, 1)


}
\keyword{models}
\keyword{regression}
