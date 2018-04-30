


\name{plot.mixing.weights}
\title{Plot mixing weights}

\alias{PlotMixingWeights}

\description{Plot the MCMC draws of the mixing weights in a \code{\link{FiniteMixture}}.}

\usage{
   PlotMixingWeights(model, style = c("boxplot", "ts", "density"),
                     burn = 0, ylim = c(0, 1), color = NULL, ...)
}

\arguments{
 \item{model}{An object of class \code{\link{FiniteMixture}}.}
 \item{style}{A string describing the desired type of plot.}
 \item{burn}{The number of MCMC iterations to be discarded before making the plot.}
 \item{ylim}{The limits on the y axis.}
 \item{color}{A vector of colors to use for the different mixture components.}
 \item{\dots}{Extra arguments passed to lower level plotting functions.}
}

\examples{
y1 <- rnorm(100, 0, 1)
y2 <- rnorm(50, 3, 1)
mix <- list(norm = NormalMixtureComponent(c(y1, y2)))
model <- FiniteMixture(mix,
                       weight.prior = DirichletPrior(rep(1, 2)),
                       niter = 100)
PlotMixingWeights(model)
plot(model, "mixing.weights")
}

\seealso{
  \code{\link{FiniteMixture}}
  \code{\link{PlotMixtureParams}}
}
