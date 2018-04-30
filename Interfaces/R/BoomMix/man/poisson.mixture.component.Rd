


\name{poisson.mixture.component}
\alias{PoissonMixtureComponent}
\title{Poisson component for finite mixtures}
\Rdversion{1.0}
\description{For modeling data as a mixture of Poisson
  distributions.}

\usage{
PoissonMixtureComponent(data, prior = NULL, group.id = NULL,
 name = "",  prior.mean = mean(data), prior.sample.size = 1, ...)
}

\arguments{

  \item{data}{A numeric vector specifying the data to be modeled.}

  \item{prior}{A prior distribution of class
    \code{\link[Boom]{GammaPrior}} for the rate (mean)
    parameter.}

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}

  \item{prior.mean}{Guess at the value of the rate parameter.  Used to
    specify the prior if \code{prior} is omitted.}

  \item{prior.sample.size}{Weight given to \code{prior.mean}.  Used to
    specify the prior if \code{prior} is omitted.}

  \item{...}{Extra arguments passed to the constructor of
    \code{\link[Boom]{GammaPrior}}, if \code{prior} is omitted.}

}

\value{An object of class \code{PoissonMixtureComponent}, which is a list
  containing information about the data and prior distribution.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{ \code{\link{FiniteMixture}}, \code{\link{BoomMix-package}}.  }
