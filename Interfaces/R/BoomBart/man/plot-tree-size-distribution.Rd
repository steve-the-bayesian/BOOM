\name{plot-tree-size-distribution}
\title{tree size distribution}
\description{Plots the number of trees in the MCMC run and the
  distribution of tree sizes.}

\alias{PlotTreeSizeDistribution}

\usage{
PlotTreeSizeDistribution(model, burn = 0)
}

\arguments{
  \item{model}{A BoomBart model object.}
  \item{burn}{The number of MCMC iterations to discard as burn-in}
}

\value{Returns \code{invisible(NULL)}.  Produces a pair of plots on the
  current graphics device.  The first shows the MCMC sample path of the
  number of trees in the ensemble.  The second shows a 7-number summary
  of the distribution of tree sizes (number of nodes in each tree).
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
x <- matrix(rnorm(5000), ncol = 10)
y <- rnorm(nrow(x), x[,1] > 0, .1)

## y is a step function of x[, 1].  The other 9 x's are irrelevant.

model <- BoomBart(y ~ x,
                  niter = 1000,
                  initial.number.of.trees = 1,
                  number.of.trees.prior = PoissonPrior(1, 1, 200))

PlotTreeSizeDistribution(model)
}
