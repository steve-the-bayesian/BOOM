\name{plot.marginal.inclusion.probabilities}
\Rdversion{1.1}
\alias{PlotMarginalInclusionProbabilities}
\title{
  Plot marginal inclusion probabilities.
}
\description{
  Produces a barplot of the marginal inclusion probabilities for a
  set of model coefficients sampled under a spike and slab prior.
  The coefficients are sorted by the marginal inclusion
  probability, and shaded by the conditional probability that a
  coefficient is positive, given that it is nonzero.
}

\usage{
PlotMarginalInclusionProbabilities(
     beta,
     burn = 0,
     inclusion.threshold = 0,
     unit.scale = TRUE,
     number.of.variables = NULL,
     ...)
}
\arguments{

  \item{beta}{ A matrix of model coefficients.  Each row represents an
     MCMC draw.  Each column represents a coefficient for a variable.
   }

  \item{burn}{
    The number of MCMC iterations in the ojbect to be discarded as
    burn-in.
  }

  \item{inclusion.threshold}{ Only plot coefficients with posterior
    inclusion probabilities exceeding this value.  }

  \item{unit.scale}{A logical value indicating whether the scale of the
    plot should be from 0 to 1.  Otherwise the scale is determined by
    the maximum inclusion probability.}

  \item{number.of.variables}{If non-\code{NULL} this specifies the
    number of coefficients to plot, taking precedence over
    \code{inclusion.threshold}.}

  \item{\dots}{
    Additional arguments to be passed to \code{\link{barplot}.}
  }

}
\value{
  Invisibly returns a list with the following elements.
  \item{barplot}{The midpoints of each bar, which is useful for adding
    to the plot.}

  \item{inclusion.prob}{The marginal inclusion probabilities of each
    variable, ordered smallest to largest (the same order as the plot).}

  \item{positive.prob}{The probability that each variable has a
    positive coefficient, in the same order as \code{inclusion.prob}.}

  \item{permutation}{The permutation of beta that puts the
    coefficients in the same order as \code{positive.prob} and
    \code{inclusion.prob}.  That is: \code{beta[, permutation]} will have the
    most significant coefficients in the right hand columns.}
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{summary.lm.spike}}
  \code{\link{predict.lm.spike}}
}

\examples{
simulate.lm.spike <- function(n = 100, p = 10, ngood = 3, niter=1000, sigma = 8){
  x <- cbind(matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  y <- rnorm(n, beta[1] + x \%*\% beta[-1], sigma)
  draws <- lm.spike(y ~ x, niter=niter)
  return(invisible(draws))
}
model <- simulate.lm.spike(n = 1000, p = 50, sigma = .3)
plot(model, inclusion.threshold = .01)
}
\keyword{models}
\keyword{regression}
