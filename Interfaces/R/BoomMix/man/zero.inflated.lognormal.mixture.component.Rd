


\name{zero.inflated.lognormal.mixture.component}
\alias{ZeroInflatedLognormalMixtureComponent}
\title{Zero-inflated lognormal component for finite mixtures}
\Rdversion{1.0}

\description{For modeling data as a mixture of zero-inflated lognormal
  distributions.  The zero-inflated lognormal is a mixture of a
  lognormal with zero.  The parameters of this model are the probability
  that an observation is positive, and the mean and variance of the log
  of the positive observations.}

\usage{
ZeroInflatedLognormalMixtureComponent(data, beta.prior = NULL,
  normal.inverse.gamma.prior = NULL, group.id = NULL,
  name = "", ...)
}

\arguments{

  \item{data}{A numeric vector specifying the data to be modeled.}

  \item{beta.prior}{An object of class
    \code{\link[Boom]{BetaPrior}} describing the prior
    distribution of the probability that a data point is nonzero.}

  \item{normal.inverse.gamma.prior}{An object of class
    \code{\link[Boom]{NormalInverseGammaPrior}} describing prior beliefs about
    the mean and variance parameters for the normal model governing the
    logs of the positive observations.}

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}

  \item{...}{Extra arguments passed to the constructors of
    \code{\link[Boom]{BetaPrior}} or
    \code{\link[Boom]{NormalInverseGammaPrior}}, if either is omitted.}

}

\value{An object of class \code{ZeroInflatedLognormalMixtureComponent},
  which is a list containing information about the data and prior
  distribution.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{ \code{\link{FiniteMixture}}, \code{\link{BoomMix-package}}.  }
