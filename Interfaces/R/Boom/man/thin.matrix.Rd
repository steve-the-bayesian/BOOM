\name{thin.matrix}
\title{Thin a Matrix}
\alias{ThinMatrix}

\description{Return discard all but every k'th row of a matrix.}

\usage{
ThinMatrix(mat, thin)
}

\arguments{
  \item{mat}{The matrix to be thinned.}
  \item{thin}{The distance between kepts lines from mat.  The larger the number
    the fewer lines are kept.  }
}

\value{
  The matrix mat, after discarding all but every \code{thin} lines.
}

\details{
  The bigger the value of \code{thin} the more thinning that gets done.
  For example, \code{thin = 10} will keep every 10 lines from \code{mat}.
}

\examples{
m <- matrix(1:100, ncol = 2)
ThinMatrix(m, thin = 10)
##      [,1] [,2]
## [1,]   10   60
## [2,]   20   70
## [3,]   30   80
## [4,]   40   90
## [5,]   50  100
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
