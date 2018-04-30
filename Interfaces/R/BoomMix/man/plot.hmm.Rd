

\name{plot.hmm}
\title{Plot a hidden Markov model}
\alias{plot.HiddenMarkovModel}
\alias{plot.hmm}
\alias{plot.hidden.markov.model}

\description{S3 method to plot the output of a hidden Markov model fit
  using \code{\link{HiddenMarkovModel}}.}

\usage{
  \method{plot}{HiddenMarkovModel}(x,
     y = c("state.probabilities",
           "transition.probabilities",
           "initial.state.distribution",
           "loglikelihood",
           "log.likelihood",
           "logprior",
           "log.prior"),
     burn = 0,
     ...)
}

\arguments{
  \item{x}{An object of class \code{\link{HiddenMarkovModel}} to plot.}
  \item{y}{A character string indicating what to plot.  The choices for
    \code{y} include those listed above, or a string that matches the
    names of one or more mixture components or parameters in \code{x}. }
  \item{burn}{The number of MCMC iterations to discard as burn-in.}
  \item{\dots}{Extra arguments passed to lower level methods.}
}

\value{This function is called for its side effect, which is to produce
  a plot on the current graphics device.}

\examples{
## Fitting a two state HMM to the classic "fetal lamb" data
data(fetal.lamb)
mix <- PoissonMixtureComponent(fetal.lamb)
model <- HiddenMarkovModel(mix, state.space.size = 2, niter = 100, ping = 10)

plot(model)                ## plot the state membership probabilities
plot(model, type = "h")    ## plot the state membership probabilities
plot(model, "lambda")      ## plot the poisson rates in each state
plot(model, "transition.probabilities")
plot(model, "trans")       ## you can shorten the names to a unique stem
plot(model, "trans", style = "box")  ## Many plots accept a 'style' argument
plot(model, "trans", style = "acf")  ## Many plots accept a 'style' argument
plot(model, "loglike")     ## A good global check for convergence
}

\seealso{
  \code{\link[BoomMix]{PlotMixtureParams}}
  \code{\link{PlotTransitionProbabilities}}
  \code{\link{PlotInitialStateDistribution}}
  \code{\link[Boom]{CompareDensities}}
  \code{\link[Boom]{CompareVectorBoxplots}}
}
