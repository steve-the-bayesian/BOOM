

\name{plot.initial.distribution}
\title{Plot HMM initial state distribution}
\alias{plot.initial.state}
\alias{plot.initial.distribution}
\alias{plot.initial.state.distribution}
\alias{PlotInitialState}
\alias{PlotInitialDistribution}
\alias{PlotInitialStateDistribution}

\description{Plot the posterior distribution of the initial state
distribution for a \code{\link{HiddenMarkovModel}}.}

\usage{
  PlotInitialStateDistribution(hmm,
                               style = c("ts", "box", "acf"),
                                burn = 0,
                                ylim = c(0, 1),
                                colors = NULL,
                                ...)
}

\arguments{
  \item{hmm}{An object of class \code{\link{HiddenMarkovModel}}.}
  \item{style}{The style of plot to produce.}
  \item{burn}{The number of MCMC iterations to discard as burn-in.}
  \item{ylim}{Limits on the vertical axis.}
  \item{colors}{A vector of colors to use for the plots.}
  \item{\dots}{Extra arguments passed to \code{\link[Boom]{PlotManyTs}},
    \code{\link[Boom]{PlotMacf}}, or \code{\link[Boom]{BoxplotMcmcMatrix}}.}

}

\value{This function is called for its side effect, which is to produce
  a plot on the current graphics device.}

\examples{
## Fitting a two state HMM to the classic "fetal lamb" data
data(fetal.lamb)
mix <- PoissonMixtureComponent(fetal.lamb)
model <- HiddenMarkovModel(mix, state.space.size = 2, niter = 100, ping = 10)

plot(model, "initial.state.distribution")
plot(model, "init")       ## you can shorten the names to a unique stem
plot(model, "init", style = "box")  ## Many plots accept a 'style' argument
plot(model, "init", style = "acf")  ## Many plots accept a 'style' argument
}

\seealso{
  \code{\link{plot.HiddenMarkovModel}}
  \code{\link[BoomMix]{PlotMixtureParams}}
  \code{\link{PlotTransitionProbabilities}}
  \code{\link[Boom]{CompareDensities}}
  \code{\link[Boom]{CompareVectorBoxplots}}
}
