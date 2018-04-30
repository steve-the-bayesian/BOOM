


\alias{PoissonProcessComponent}
\name{poisson.process.component}
\title{Components of the Markov modulated Poisson process}
\Rdversion{1.0}

\description{
  Specify a component of a \code{\link{MarkovModulatedPoissonProcess}}.
}

\usage{
PoissonProcessComponent(process,
                        spawns = character(0),
                        kills = character(0),
                        mixture.component = character(0))
}

\arguments{

  \item{process}{An object inheriting from the class
     \code{\link{PoissonProcess}}, representing the PoissonProcess
     component to be added.}

  \item{spawns}{A character vector naming the processes that are
     activated by an event from \code{process}.}

  \item{kills}{A character vector naming the processes that are
     deactivated by an event from \code{process}.}

  \item{mixture.component}{A string naming the mixture component that
     models the marks for \code{process}.  The definition of the mixture
     component model family takes place in the call to
     \code{\link{MarkovModulatedPoissonProcess}}.}

 }

\value{ Checks the function arguments for validity, and returns an
  object of class \code{PoissonProcessComponent}, which is a list
  containing the values of the function arguments.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
## Please see the example in ?MarkovModulatedPoissonProcess.
}

\seealso{
  \code{\link{MarkovModulatedPoissonProcess}}, \code{\link{PoissonProcess}},
  \code{\link{BoomEvents-package}}.
}

\keyword{models}
