


\alias{SimulateWeeklyCyclePoissonProcess}
\name{simulate.weekly.cycle.poisson.process}
\title{Simulate a Poisson process with a weekly cycle}
\Rdversion{1.0}

\description{
  Simulates events from a Poisson process with a weekly cycle
  parameterized like \code{\link{WeeklyCyclePoissonProcess}}.
}

\usage{
  SimulateWeeklyCyclePoissonProcess(start, end,
                                    lambda,
                                    daily,
                                    weekday.hourly, weekend.hourly)
}

\arguments{

  \item{start}{A time convertible to \code{\link{POSIXct}}.  The
    beginning of the observation window.}

  \item{end}{A time convertible to \code{\link{POSIXct}}.  The end of the
    observation window.}

  \item{lambda}{The average daily number of events.}

  \item{daily}{A vector of non-negative numbers that gives the each
       day's multiplicative effect on the event rate.  The vector must
       have 7 elements and sum to 7.}

  \item{weekday.hourly}{A vector of non-negative numbers giving each
       hour's effect on the event rate.  The vector must have 24
       elements and sum to 24.  Describes the hourly effect on a week
       day.}

  \item{weekend.hourly}{A vector of non-negative numbers giving each
       hour's effect on the event rate.  The vector must have 24
       elements and sum to 24.  Describes the hourly effect on a weekend
       day.}

}

\value{An object of class \code{\link{PointProcess}} with the specified
  weekly profile.
}


\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomEvents-package}}.
}

\examples{

lambda <- 12
delta <- runif(7)
delta <-  7 * delta / sum(delta)
eta.weekend <- runif(24)
eta.weekend <- 24 * eta.weekend / sum(eta.weekend)
eta.weekday <- runif(24)
eta.weekday <- 24 * eta.weekday / sum(eta.weekday)
result <- SimulateWeeklyCyclePoissonProcess(
            "2008-04-17", "2008-05-12", lambda, delta, eta.weekday, eta.weekend)
plot(result)
plot(WeeklyProfile(result))
}

\keyword{models}
