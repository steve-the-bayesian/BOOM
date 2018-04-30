

\name{plot.transition.probabilities}
\title{Plot HMM transition probabilities}
\alias{plot.transition.probabilities}
\alias{PlotTransitionProbabilities}

\description{Plot the posterior distribution of the transition
  probabilities for a \code{\link{HiddenMarkovModel}}.}

\usage{
  PlotTransitionProbabilities(hmm,
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

plot(model, "transition.probabilities")
plot(model, "trans")       ## you can shorten the names to a unique stem
plot(model, "trans", style = "box")  ## Many plots accept a 'style' argument
plot(model, "trans", style = "acf")  ## Many plots accept a 'style' argument
}

\seealso{
  \code{\link{plot.HiddenMarkovModel}}
  \code{\link[BoomMix]{PlotMixtureParams}}
  \code{\link{PlotInitialStateDistribution}}
  \code{\link[Boom]{CompareDensities}}
  \code{\link[Boom]{CompareVectorBoxplots}}
}
