\name{plot.coefficients}
\Rdversion{1.1}
\alias{PlotLmSpikeCoefficients}
\title{
  Plot Coefficients.
}

\description{
  Produces boxplots showing the marginal distribution of the coefficients.
}

\usage{
PlotLmSpikeCoefficients(
     beta,
     burn = 0,
     inclusion.threshold = 0,
     scale.factors = NULL,
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

  \item{scale.factors}{If non-null then a vector of scale factors with which to
    scale the columns of beta.  A \code{NULL} value is ignored.}

  \item{number.of.variables}{If non-\code{NULL} this specifies the
    maximum number of coefficients to plot.  A \code{NULL} value is
    ignored.}

  \item{\dots}{
    Additional arguments to be passed to \code{\link{boxplot}.}
  }

}
\value{
  Returns the value from the final call to \code{boxplot}.
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
simulate.lm.spike <- function(n = 100, p = 10, ngood = 3, niter=1000, sigma = 1){
  x <- cbind(matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  y <- rnorm(n, beta[1] + x \%*\% beta[-1], sigma)
  draws <- lm.spike(y ~ x, niter=niter)
  return(invisible(draws))
}
model <- simulate.lm.spike(n = 1000, p = 50, sigma = .3)
plot(model, "coef", inclusion.threshold = .01)
}
\keyword{models}
\keyword{regression}
