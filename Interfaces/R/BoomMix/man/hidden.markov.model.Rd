


\alias{HiddenMarkovModel}
\alias{HMM}
\alias{hmm}
\name{hidden.markov.model}
\title{MCMC for hidden Markov models}
\Rdversion{1.0}

\description{
  Uses MCMC (data augmentation) to sample from the posterior
  distribution of a hidden Markov model.
}

\usage{
HiddenMarkovModel(mixture.components, state.space.size,
                  markov.model.prior = NULL,
                  niter, ping = niter/10, seed = NULL)
}

\arguments{

  \item{mixture.components}{Either a single object, or a list of
    objects, inheriting from class \code{MixtureComponent}.  See
    \code{\link[BoomMix]{BoomMix-package}} for comments on how to
    structure mixture components.}

  \item{state.space.size}{The number of states in the hidden Markov
     chain.  If \code{markov.model.prior} is specified then this
     argument is ignored and its value is inferred from
     \code{markov.model.prior}.}

   \item{markov.model.prior}{An object of class
     \code{\link[Boom]{MarkovPrior}} giving the prior
     distribution of the hidden Markov chain.}

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{ The frequency of status update messages.  If ping > 0
    then the program will print a status message to the screen every
    \code{ping} MCMC iterations.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}
}

\value{ An object of class HiddenMarkovModel, which is a list containing
  the following elements:

  \item{transition.probabilities}{A 3-dimensional array containing the
    MCMC draws of the transition probability matrix for the hidden
    Markov chain.  The first dimension is the MCMC iteration.  The
    second and third are the current and next states involved in a
    transition.  Thus \code{transition.probabilities[i, , ]} is a matrix
    with rows that sum to 1. }

  \item{ParameterName.StateNumber}{MCMC draws for the parameters of all
    the mixture components.  The names of each parameter are taken from
    the type of mixture component, the name it was assigned (either
    directly or in \code{mixture.components}), and the state number.  In
    each case the parameter is an array with one extra dimension (the
    first) corresponding to MCMC iteration.}

  \item{loglike}{The log-likelpihood associated with each MCMC draw.}

  \item{logprior}{The log prior density associated with each MCMC draw.}

  \item{state.probabilities}{Marginal state membership probabilities for
       each data point used to fit the model.  If \code{group.id} was
       supplied to the mixture components then this is a list of
       matrices, with each matrix corresponding to a group.  Otherwise
       it is a single matrix.  In either case, row \code{i} of a matrix
       corresponds to the observation at time \code{i} and each column
       to a latent state.}

}


\references{

  Fruhwirth-Schnatter (2006), "Finite mixture and Markov switching models", Springer.

  Scott (2002), "Bayesian methods for hidden Markov models:  Recursive
  computing in the 21st century". \emph{JASA} pp 337--351.

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link[BoomMix]{BoomMix-package}}.
}

\examples{
## Fitting a two state HMM to the classic "fetal lamb" data
data(fetal.lamb)
mix <- PoissonMixtureComponent(fetal.lamb)
model <- HiddenMarkovModel(mix, state.space.size = 2, niter = 1000, ping = 100)
plot(model$lambda.0, model$lambda.1, main = "Poisson Rates")

## Compre the clustering you get from an HMM to a finite mixture model.
## Simulate some data, a change point occurs at observation 100 ->
## 101. The data are multivariate (one Gaussian and one Poisson variable).
y1 <- rnorm(100, 0, 1)
z1 <- rpois(100, 1)
y2 <- rnorm(50, 3, 1)
z2 <- rpois(50, 3)

norm.mix <- NormalMixtureComponent(c(y1, y2))
pois.mix <- PoissonMixtureComponent(c(z1, z2))

mixture.model <- FiniteMixture(list(normal = norm.mix, poisson = pois.mix),
                               state.space.size = 2, niter = 1000)
hmm <- HiddenMarkovModel(list(normal = norm.mix, poisson = pois.mix),
                               state.space.size = 2, niter = 1000)

opar <- par(mfrow = c(1,2))
plot(mixture.model$state.probabilities[,1],
     type = "h",
     main = "MixtureModel",
     xlab = "Observation Number",
     ylab = "State Probability (labelling is arbitrary)")

plot(hmm$state.probabilities[,1],
     type = "h",
     main = "Hidden Markov Model",
     xlab = "Observation Number",
     ylab = "State Probability (labelling is arbitrary)")
par(opar)
}
\keyword{models}
