\name{dirichlet-process}
\alias{DirichletProcessMvn}
\title{Dirichlet Process Mixtures of Multivariate Normals}
\Rdversion{1.0}
\description{
  Fits a Dirichlet process mixtures of multivariate normal distributions
  to a data set using MCMC.
}

\usage{
DirichletProcessMvn(data,
                   mean.base.measure = NULL,
                   variance.base.measure = NULL,
                   concentration.parameter = 1,
                   niter,
                   ping = niter / 10,
                   seed = NULL)
}

\arguments{

  \item{data}{A numeric matrix of data, or a data frame convertible to
    such a matrix.}

  \item{mean.base.measure}{An object of class
    \code{\link[Boom]{MvnGivenSigmaMatrixPrior}} from the Boom library.  This
    is a probability distribution describing the means of the mixture
    components.  If \code{NULL} then a default value will be used.}

  \item{variance.base.measure}{An object of class
    \code{\link{InverseWishartPrior}} from the Boom library.  If
    \code{NULL} then a default value will be used.}

  \item{concentration.parameter}{The concentration parameter for the
    Dirichlet process. A positive scalar.  Larger values lead to more
    mixture components.}

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{ The frequency of status update messages.  If ping > 0
    then the program will print a status message to the screen every
    \code{ping} MCMC iterations.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}
}

\value{
  The returned object is a list imbued wtih class "DirichletProcessMvn".
  There are two list elements.

  \itemize{
    \item{\code{log likelihood} gives the log likelihood associated with each MCMC draw.
      This is primarily used as a convergence diagnostic.
    }
    \item {parameters is a list of the drawn parameter values.  The list elements
      are named according to the number of clusters that were used in each
      draw.  The cluster sizes are arranged in increasing order, but need
      not be congiguous.  The elements of the 'parameters' list are:
      \itemize{
	\item{mean: A 3-way array of draws of cluster means.  The array dimensions
	  are [Monte Carlo index, cluster number, data dimension].}
	\item{variance: A 4-way array of draws of cluster variances.  The array
	  dimensions are [Monte Carlo index, cluster number, data dimension,
	  data dimension].}
	\item{iteration: A vector containing the draw number of each draw in the
	  overall Monte Carlo sequence.  This information is needed so that
	  burn-in iterations can be discarded.}
      }
    }
  }
}

\references{

  Fruhwirth-Schnatter (2006), "Finite mixture and Markov switching models", Springer.

  McLachlan and Peel (2000) "Finite Mixture Models", Wiley.

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomMix-package}}.
}

\examples{
y0 <- rmvn(n=100, mu = 1:3)
y1 <- rmvn(n = 200, mu = 3:1)
y2 <- rmvn(n = 150, mu = rep(3, 3))
y <- rbind(y0, y1, y2)
model <- DirichletProcessMvn(y, niter=1000, seed=8675309)
plot(model, "nclusters")
plot(model, nclusters=3)
}
\keyword{models}
\keyword{regression}
