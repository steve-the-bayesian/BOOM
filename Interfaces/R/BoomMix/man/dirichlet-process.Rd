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

  An object of class \code{FiniteMixture}, which is a list with
  components for mixing weights, and for the parameters of the mixture
  components.  Each paramter is stored in an array matching the
  dimension of the parameter, plus 1.  Thus scalars are stored in
  vectors.  Vectors are stored in matrices, and matrices are stored in
  3-way arrays.  The extra dimension (always the first in each array)
  corresponds to MCMC iteration.  }

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
y1 <- rnorm(100, 0, 1)
y2 <- rnorm(50, 3, 1)
z1 <- rpois(100, 3)
z2 <- rpois(50, 6)
norm.mix <- NormalMixtureComponent(c(y1, y2))
pois.mix <- PoissonMixtureComponent(c(z1, z2))
model <- FiniteMixture(list(norm.mix, pois.mix),
                       weight.prior = DirichletPrior(rep(1, 2)),
                       niter = 100,
                       ping = 10)
}
\keyword{models}
\keyword{regression}
