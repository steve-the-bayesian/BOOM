\name{ToString}
\title{Convert to Character String}
\alias{ToString}
\alias{ToString.default}
\alias{ToString.table}

\description{Convert an object to a character string, suitable for
  including in error messages.}

\usage{
  ToString(object, ...)
}

\arguments{
  \item{object}{An object to be printed to a string.}
  \item{\dots}{Extra arguments passed to \code{\link{print}}.}
}

\value{
  A string (a character vector of length 1) containing the printed value
  of the object.
}

\examples{
  m <- matrix(1:6, ncol = 2)
  printed.matrix <- ToString(m)

  y <- c(1, 2, 3, 3, 3, 3, 3, 3)
  tab <- table(y)
  printed.table <- ToString(tab)
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
