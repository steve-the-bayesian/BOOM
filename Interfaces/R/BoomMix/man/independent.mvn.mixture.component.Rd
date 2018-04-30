


\name{independent.mvn.mixture.component}

\alias{IndependentMvnMixtureComponent}
\title{Multivariate normal component for finite mixtures}
\Rdversion{1.0}

\description{For modeling data as a mixture of multivariate Gaussian
  distributions.  This differs from \code{\link{MvnMixtureComponent}}
  because the variables are assumed to all be independent of one another
  (i.e. off-diagonal covariances are constrained to be zero).  This
  component is intended for cases where there are a very large number of
  variables, making it difficult to estimate the variance matrix in
  an \code{\link{MvnMixtureComponent}}.}

\usage{
IndependentMvnMixtureComponent(data,
                               prior.mean.guess = apply(data, 2, mean),
                               prior.mean.sample.size = 1.0,
                               prior.sd.guess = apply(data, 2, sd),
                               prior.sd.sample.size = 1.0,
                               sigma.upper.limit = NULL,
                               group.id = NULL,
                               name = "")
}

\arguments{

  \item{data}{A matrix specifying the data to be modeled.  Each row is
    an observation.}

  \item{prior.mean.guess}{ A vector giving the prior mean for each
    variable in data.}

  \item{prior.mean.sample.size}{A vector giving the weight, in terms of
    prior observations, to be assigned to \code{prior.mean.guess}.}

  \item{prior.sd.guess}{A vector giving a guess at the standard
    deviation for each variable in data.}

  \item{prior.sd.sample.size}{A vector giving the weight, in therms of
    prior observations, to be assigned to \code{prior.sd.guess}.}

  \item{sigma.upper.limit}{A vector giving the maximum acceptable value
    of the standard deviation for each variable in data.  If \code{NULL}
    then all upper limits will be assumed infinite.  }

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}
}

\value{An object of class \code{IndependentMvnMixtureComponent}, which
  is a list containing information about the data and prior
  distribution.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
z1 <- matrix(rnorm(100, 1:10, 1), ncol = 10, byrow = TRUE)
z2 <- matrix(rnorm(200, 10:1, 1), ncol = 10, byrow = TRUE)
mix <- IndependentMvnMixtureComponent(rbind(z1, z2))
model <- FiniteMixture(mix, 2, niter = 500)
}


\seealso{
  \code{\link{FiniteMixture}},
  \code{\link{BoomMix-package}},
  \code{\link{MvnMixtureComponent}}.
}
