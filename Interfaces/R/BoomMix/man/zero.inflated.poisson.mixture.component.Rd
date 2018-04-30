


\name{zero.inflated.poisson.mixture.component}
\alias{ZeroInflatedPoissonMixtureComponent}
\title{Zero-inflated Poisson component for finite mixtures}
\Rdversion{1.0}

\description{For modeling data as a mixture of zero-inflated Poisson
  distributions.  The zero-inflated Poisson is a mixture of a Poisson
  distribution with a point mass at zero.  The parameters of this model
  are the probability that an observation is a forced zero, and the mean
  of the Poisson observations (some of which might be zero).}

\usage{

ZeroInflatedPoissonMixtureComponent(data, lambda.prior = NULL,
    zero.probability.prior = NULL, group.id = NULL, name = "")
}

\arguments{

  \item{data}{A numeric vector specifying the data to be modeled.}

  \item{lambda.prior}{An object of class GammaPrior, where the first
    (shape, or 'alpha') parameter corresponds to a sum of prior
    events, and the second (scale, or 'beta') parameter
    corresponds to a prior sample size.
  }

  \item{zero.probability.prior}{An object of class BetaPrior to use as
    the prior for the probability of a forced zero. }

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}

}

\value{An object of class \code{ZeroInflatedPoissonMixtureComponent},
  which is a list containing information about the data and prior
  distribution.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
## With no forced zeros you should see zero.probability with most of its
## mass near zero, and lambda.0 concentrated around 3.
y <- rpois(100, 3)
mix <- ZeroInflatedPoissonMixtureComponent(y)
model <- FiniteMixture(mix, 1, niter = 1000)
opar <- par(mfrow = c(1,2))
hist(model$lambda.0)
hist(model$zero.probability.0)

## Now add in some forced zeros.
y <- c(rep(0, 20), y)
## The distribution of lambda should not change very much, but the
## distribution of zero.probability should be near 1/6 = .16667.
mix <- ZeroInflatedPoissonMixtureComponent(y)
model <- FiniteMixture(mix, 1, niter = 1000)
par(mfrow = c(1,2))
hist(model$lambda.0)
hist(model$zero.probability.0)
par(opar)
}

\seealso{ \code{\link{FiniteMixture}}, \code{\link{BoomMix-package}}.  }
