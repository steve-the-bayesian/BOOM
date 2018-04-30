


\name{mvn.mixture.component} \alias{MvnMixtureComponent}
\title{Multivariate normal component for finite mixtures}
\Rdversion{1.0} \description{For modeling data as a mixture of
  multivariate Gaussian distributions.}

\usage{
MvnMixtureComponent(data, prior = NULL, group.id = NULL,
 name = "", ...)
}

\arguments{

  \item{data}{A matrix specifying the data to be modeled.  Each row is
    an observation.}

  \item{prior}{A prior distribution of class
    \code{\link[Boom]{NormalInverseWishartPrior}} for the mean
    and variance of the mixture component.}

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}

  \item{...}{Extra arguments passed to the constructor of
    \code{\link[Boom]{NormalInverseWishartPrior}}, if \code{prior} is omitted.}
}

\value{An object of class \code{MvnMixtureComponent}, which is a list
  containing information about the data and prior distribution.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{ \code{\link{FiniteMixture}}, \code{\link{BoomMix-package}}.  }
