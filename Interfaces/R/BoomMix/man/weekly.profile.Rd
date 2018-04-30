


\alias{WeeklyProfile}
\alias{plot.WeeklyProfile}
\alias{print.WeeklyProfile}
\name{weekly.profile}
\title{Weekly profile of point process data}
\Rdversion{1.0}

\description{
  Describes the weekly seasonal pattern in point process data in terms
  of the number of events per day and per hour.
}

\usage{
  WeeklyProfile(timestamps)
  \method{print}{WeeklyProfile}(x, ...)
  \method{plot}{WeeklyProfile}(x, ...)
}

\arguments{

  \item{timestamps}{This can either be a vector of time points coercible to
    \code{\link{POSIXt}}, or an object of class
    \code{\link{PointProcess}}.  }

  \item{x}{An object of class \code{WeeklyProfile} to plot or print.}

  \item{\dots}{Extra arguments (which are ignored).}
}

\value{

    A \code{WeeklyProfile} object containing summaries of the weekly
    seasonal pattern in \code{timestamps}.  The summaries include counts
    of the number of times an event occurred on each day of the week and
    on each hour of the day.  The hourly counts are split by weekends
    and weekdays.

    The \code{plot} method plots the counting process for the supplied
    timestamps and produces barplots for the daily and weekly effects.

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomEvents-package}}.
}

\examples{
\dontrun{
lambda <- 12
delta <- runif(7)
delta <-  7 * delta / sum(delta)
eta.weekend <- runif(24)
eta.weekend <- 24 * eta.weekend / sum(eta.weekend)
eta.weekday <- runif(24)
eta.weekday <- 24 * eta.weekday / sum(eta.weekday)
result <- SimulateWeeklyCyclePoissonProcess(
            "2008-04-17", "2008-05-12", lambda, delta, eta.weekday, eta.weekend)
plot(WeeklyProfile(result))
}
}

\keyword{models}
