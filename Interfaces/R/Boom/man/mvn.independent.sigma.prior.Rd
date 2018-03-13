\name{mvn.independent.sigma.prior}
\alias{MvnIndependentSigmaPrior}
\title{Independence prior for the MVN}

\description{ A prior for the parameters of the multivariate normal
  distribution that assumes Sigma to be a diagonal matrix with elements
  modeled by independent inverse Gamma priors.  }


\usage{
MvnIndependentSigmaPrior(mvn.prior, sd.prior.list)
}

\arguments{
  \item{mvn.prior}{An object of class \code{\link{MvnPrior}} that is the
    prior distribution for the multivariate normal mean parameter.}

  \item{sd.prior.list}{A list of \code{\link{SdPrior}} objects modeling
    the diagonal elements of the multivariate normal variance matrix.  The
    off-diagonal elements are assumed to be zero.}
}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
