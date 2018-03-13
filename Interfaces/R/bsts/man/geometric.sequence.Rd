\name{geometric.sequence}

\alias{GeometricSequence}

\Rdversion{1.0}

\title{
  Create a Geometric Sequence
}

\description{ Create a geometric sequence.}

\usage{
  GeometricSequence(length, initial.value = 1, discount.factor = .5)
}

\arguments{

  \item{length}{A positive integer giving the length of the desired sequence.}

  \item{initial.value}{The first term in the sequence.  Cannot be zero.}

  \item{discount.factor}{The ratio between a sequence term and the
    preceding term.  Cannot be zero.}

}

\value{
  A numeric vector containing the desired sequence.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
GeometricSequence(4, .8, .6)
# [1] 0.8000 0.4800 0.2880 0.1728

GeometricSequence(5, 2, 3)
# [1]   2   6  18  54 162

\dontrun{
GeometricSequence(0, -1, -2)
# Error: length > 0 is not TRUE
}
}
