


\alias{PoissonProcess}
\alias{HomogeneousPoissonProcess}
\alias{WeeklyCyclePoissonProcess}
\name{poisson.process}
\title{Specify a Poisson process model}
\Rdversion{1.0}

\description{
  Specify a model of class PoissonProcess that can be used as part of a
  \code{\link{MarkovModulatedPoissonProcess}} model.
}

\usage{
  HomogeneousPoissonProcess(lambda.prior)
  WeeklyCyclePoissonProcess(average.daily.rate.prior,
                            daily.dirichlet.prior,
                            weekday.hourly.dirichlet.prior,
                            weekend.hourly.dirichlet.prior)
}

\arguments{

  \item{lambda.prior}{An object of class \code{\link{GammaPrior}}
       specifying the prior distribution on the Poisson rate parameter.
       The first (shape, alpha) argument to the GammaPrior is the prior
       number of observed events.  The second (scale, beta) is the prior
       exposure time, which is measured in days.}

  \item{average.daily.rate.prior}{An object of class
       \code{\link{GammaPrior}} giving the prior distribution on the
       average daily rate of the process.  The first (shape, alpha)
       argument to the GammaPrior is the prior number of observed
       events.  The second (scale, beta) is the prior exposure time,
       which is measured in days.}

  \item{daily.dirichlet.prior}{An object of class
    \code{\link{DirichletPrior}} giving the prior distribution of the
    day of week effects.}

  \item{weekday.hourly.dirichlet.prior}{An object of class
       \code{\link{DirichletPrior}} giving the prior distribution of the
       hour-of-day effects for non-weekend days.  }

  \item{weekend.hourly.dirichlet.prior}{An object of class
       \code{\link{DirichletPrior}} giving the prior distribution of the
       hour-of-day effects for weekend days.}

 }

\value{ The returned objects inherit from the \code{PoissonProcess} base
  class, so they can be used to define the specification for a
  \code{\link{MarkovModulatedPoissonProcess}} model.

  The \code{HomogeneousPoissonProcess} assumes a homogeneous rate.

  The \code{WeeklyCyclePoissonProcess} assumes a rate function that is
  constant within each hour of the week.  The rate function is
  decomposed as rate(day, hour) = lambda * delta[day] * eta[hour].  The
  sum of delta is 7 and the sum of eta is 24, so lambda can be
  interpreted as the long run average number of events per day.  A
  different eta applies on weekends and weekdays.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomEvents-package}}.
}

\keyword{models}
