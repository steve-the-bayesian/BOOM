\name{thin}
\Rdversion{1.1}
\alias{thin}
\title{
  Thin the rows of a matrix
}
\description{
  Systematic sampling of every \code{thin}'th row of a matrix or vector.
  Useful for culling MCMC output or denoising a plot.
}
\usage{
thin(x, thin)
}

\arguments{
  \item{x}{
    The array to be thinned.  The first dimension is the one sampled over.
}
  \item{thin}{
    The frequency of observations to keep.  With \code{thin=10} you will
  keep every 10th observation.
}
}
\value{
  The thinned vector, matrix, or array is returned.
}
\author{
  Steven L. Scott
}

\examples{

x <- rnorm(100)
thin(x, 10)
# returns a 10 vector

y <- matrix(rnorm(200), ncol=2)
thin(y, 10)
# returns a 10 by 2 matrix

}
\keyword{ dplot }
