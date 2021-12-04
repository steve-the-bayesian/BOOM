


\name{plot.finite.mixture}
\title{Plot a finite mixture model}

\alias{plot.FiniteMixture}
\alias{plot.mixture}
\alias{plot.finite.mixture}

\description{S3 method to plot the output of a finite mixture model fit
  using \code{\link{FiniteMixture}}.}

\usage{
  \method{plot}{FiniteMixture}(x,
     y = c("state.probabilities", "mixing.weights",
           "loglikelihood", "log.likelihood",
           "logprior", "log.prior", "density"),
     burn = 0,
     ...)

}

\arguments{
  \item{x}{An object of class \code{\link{FiniteMixture}}.}

  \item{y}{A character string indicating what to plot.  The choices for
    \code{y} include those listed above, or a string that matches the
    names of one or more mixture components or parameters in \code{x}.  }

  \item{burn}{The number of MCMC iterations to discard as burn-in.}

  \item{\dots}{Extra arguments passed to lower level methods.}

}

\value{ This function is called for its side effect, which is to produce
  a plot on the current graphics device.  }

\examples{
y1 <- rnorm(100, 0, 1)
y2 <- rnorm(50, 3, 1)
z1 <- rpois(100, 3)
z2 <- rpois(50, 6)
norm.mix <- NormalMixtureComponent(c(y1, y2))
pois.mix <- PoissonMixtureComponent(c(z1, z2))
model <- FiniteMixture(list(normal = norm.mix, poisson = pois.mix),
                       2, niter = 100, ping = 10)

plot(model)
plot(model, "normal", burn = 20)
plot(model, "normal", style = "ts")
plot(model, "poisson")
plot(model, "mixing.weights")
}

\seealso{
  \code{\link{PlotDensityDistribution}}
  \code{\link{PlotMixtureParams}}
  \code{\link{PlotMixingWeights}}
  \code{\link{PlotStateProbabilities}}
  \code{\link[Boom]{CompareDensities}}
  \code{\link[Boom]{CompareVectorBoxplots}}
}
