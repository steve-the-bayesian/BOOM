\alias{CheckMixtureComponent}
\name{check.mixture.component}
\title{Error checking for mixture component specifications}
\Rdversion{1.0}

\description{A function similiar to \code{\link{stopifnot}}, but
  specialized to give better error messages when checking the structure
  of a mixture component specification.}

\usage{
  CheckMixtureComponent(..., fun = "", formula = "", data = "")
}

\arguments{

  \item{\dots}{A set of conditions that must be satisfied for the
    mixture component to be valid.}

  \item{fun}{The name of the function doing the check.  This is intended
    to be the constructor function for the mixture component.  }

  \item{formula}{If the mixture component contains a formula
    (e.g. regression and logistic regression) it can be specified to
    better identify the component that had the problem.}

  \item{data}{The name of the \code{data} argument passed to
    \code{fun}.}

}

\details{ If the conditions specified in \code{\dots} are satisfied then
  nothing happens.  Otherwise \code{stop()} is called with an error
  message explaining which condition failed, and giving any information
  provided by \code{fun}, \code{formula}, and \code{data}.  }


\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomMix-package}}.
}

\examples{
foo <- function(x, data = bar) {
  data.arg <- deparse(substitute(data))
  CheckMixtureComponent(x > 0, x < 1, fun = "foo", data = data.arg)
}

baz <- foo(.5)
## baz <- foo(2)   # raises an error with an informative message.
}

\keyword{models}
