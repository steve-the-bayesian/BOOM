\name{is.even}
\Rdversion{1.0}
\alias{is.even}
\alias{is.odd}
\alias{IsEven}
\alias{IsOdd}
\title{
  Check whether a number is even or odd.
}
\description{
  Check whether a number is even or odd.
}
\usage{
IsEven(x)
IsOdd(x)
}

\arguments{
  \item{x}{ An integer or vector of integers. }
}

\value{
  Logical indicating whether the argument is even (or odd).
}
\author{
  Steven L. Scott
}

\examples{
IsEven(2) ## TRUE
IsOdd(2)  ## FALSE
}
