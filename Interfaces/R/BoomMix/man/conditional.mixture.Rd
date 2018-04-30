\alias{ConditionalMixture}
\name{conditional.mixture}
\title{MCMC for conditional mixture models}
\Rdversion{1.0}
\description{
  Uses MCMC (data augmentation) to sample from the posterior
  distribution of a conditional mixture model.  A conditional mixture
  model is an ordinary finite mixture model where the mixing
  distribution is a multinomial logistic regression.
}

\usage{
ConditionalMixture(mixing.distribution.formula,
                   mixture.component.specification,
                   state.space.size = NULL,
                   mixing.distribution.prior = NULL,
                   niter,
                   ping = niter / 10,
                   known.source = NULL,
                   data,
                   contrasts = NULL,
                   expected.model.size = 1.0 * state.space.size,
                   seed = NULL)
}

\arguments{

  \item{mixing.distribution.formula}{A formula for the predictors of the
    mixing distribution.  The formula should not contain a response
    variable.}

  \item{mixture.component.specification}{Either a single object, or a
    list of objects, inheriting from class \code{MixtureComponent}.  See
    \code{\link{BoomMix-package}} for comments on how to structure
    mixture components.}

  \item{state.space.size}{The number of states in the mixture.  If
    \code{mixing.distribution.prior} is specified then this argument is
    ignored.}

  \item{mixing.distribution.prior}{An object of class
     \code{SpikeSlabPrior} determining the prior on the coefficients of
     the mixing weights.  The dimension of the prior must be an integer
     multiple (\code{state.space.size - 1}) of the dimension of the
     design matrix for the mixture components.  Alternatively,
     \code{mixing.distribution.prior} can be left \code{NULL}, and a
     default prior will be constructed using \code{expected.model.size}
     and \code{state.space.size}.}

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

  \item{data}{An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing
    the variables in the model.  If not found in \code{data}, the
    variables are taken from \code{environment(formula)}, typically the
    environment from which \code{lm.spike} is called.  }

  \item{contrasts}{ An optional list. See the \code{contrasts.arg}
    argument of \code{\link{model.matrix.default}}.  }

  \item{expected.model.size}{A positive number less than the number of
    elements in the vector of the multinomial logistic regression
    coefficients in the mixing distribution.  The
    \code{expected.model.size} represents a guess at the number of
    significant predictor variables.  It is used to obtain the 'spike'
    portion of the spike and slab prior.  }

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}
}

\value{

  An object of class \code{ConditionalMixture}, which is a list with
  components for the parameters of each mixture component.  Each
  parameter is stored in an array matching the dimension of the
  parameter, plus 1.  Thus scalars are stored in vectors.  Vectors are
  stored in matrices, and matrices are stored in 3-way arrays.  The
  extra dimension (always the first in each array) corresponds to MCMC
  iteration.

  The returned object also contains the values of log likelihood and log
  prior associated with each parameter draw, as well as the matrix of
  state probabilities, which gives the probability (averaging over the
  MCMC) that each observation belongs to each mixure component.  Rows in
  this matrix correspond to observations, and columns to mixture
  components.}

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
n <- 1000
x <- sort(rnorm(n, 0, 1))
f <- function(x){x * (x-2) * (x+2)}
y <- rnorm(n, f(x), 1)

mix <- RegressionMixtureComponent(y ~ x)
model <- ConditionalMixture( ~ x, mix, state.space.size = 5, niter = 100)
plot(x, y, col =max.col(model$state.probabilities))

}
\keyword{models}
\keyword{regression}
