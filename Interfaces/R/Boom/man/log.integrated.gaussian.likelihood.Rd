\name{log.integrated.gaussian.likelihood}
\title{Log Integrated Gaussian Likelihood}
\alias{LogIntegratedGaussianLikelihood}

\description{Compute the log of the integrated Gaussian likelihood,
  where the model paramaters are integrated out with respect to a
  normal-inverse gamma prior.}

\usage{
LogIntegratedGaussianLikelihood(suf, prior)
}

\arguments{
  \item{suf}{A \code{\link{GaussianSuf}} object describing the data.}

  \item{prior}{A \code{\link{NormalInverseGammaPrior} describing the
  prior distribution.}
}
}

\value{
  Returns a scalar giving the log integrated likelihood.
}


\examples{

prior <- NormalInverseGammaPrior(10, 2, sqrt(2), 1)
y <- c(7.8949, 9.17438, 8.38808, 5.52521)
suf <- GaussianSuf(y)
loglike <- LogIntegratedGaussianLikelihood(suf, prior)
# -9.73975

}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
