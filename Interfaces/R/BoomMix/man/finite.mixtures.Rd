\alias{FiniteMixture}
\name{finite.mixture}
\title{MCMC for finite mixture models}
\Rdversion{1.0}
\description{
  Uses MCMC (data augmentation) to sample from the posterior
  distribution of a finite mixture model.
}

\usage{
FiniteMixture(mixture.component.specification,
              state.space.size = NULL,
              weight.prior = NULL,
              niter,
              ping = niter / 10,
              known.source = NULL,
              seed = NULL)
}

\arguments{

  \item{mixture.component.specification}{Either a single object, or a list of
    objects, inheriting from class \code{MixtureComponent}.  See
    \code{\link{BoomMix-package}} for comments on how to structure
    mixture components.}

  \item{state.space.size}{The number of states in the mixture chain.  If
    \code{weight.prior} is specified then this argument is ignored and
    its value is inferred from \code{weight.prior}.}

  \item{weight.prior}{An object of class
    \code{\link[Boom]{DirichletPrior}} giving the prior
    distribution for the mixing weights.  The dimension of this object
    determines the number of mixture components.}

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{ The frequency of status update messages.  If ping > 0
    then the program will print a status message to the screen every
    \code{ping} MCMC iterations.}

  \item{known.source}{An optional numeric vector indicating which
    mixture component individual observations belong to.  In a typical
    finite mixture problem this information will be unavailable.  If it
    is fully available then the finite mixture model turns into a naive
    Bayes classifier.  If the components for only a few observations are
    known then the unknown ones can be marked with \code{NA}, in which
    case the model becomes a semi-supervised learner.}

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
