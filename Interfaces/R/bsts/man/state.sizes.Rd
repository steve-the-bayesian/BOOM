\name{state.sizes}

\alias{StateSizes}

\Rdversion{1.0}

\title{
  Compute state dimensions
}

\description{ Returns a vector containing the size of each state
  component (i.e. the state dimension) in the state vector.  }

\usage{
  StateSizes(state.specification)
}

\arguments{
  \item{state.specification}{A list containing state specification
  components, such as would be passed to \code{\link{bsts}}.}
}

\value{
  A numeric vector giving the dimension of each state component.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
    y <- rnorm(1000)
    state.specification <- AddLocalLinearTrend(list(), y)
    state.specification <- AddSeasonal(state.specification, y, 7)
    StateSizes(state.specification)
}

\keyword{chron}
