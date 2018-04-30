\name{number-of-trees}
\title{number of Bart trees}
\description{The number of trees used by the Bart algorithm.}
\alias{NumberOfTrees}

\usage{
NumberOfTrees(model)
}

\arguments{
  \item{model}{A \code{\link{BoomBart}} model object.}
}

\value{
  A numeric vector giving the number of trees in the model ensemble
  after each MCMC iteration.
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

ntrees <- NumberOfTrees(model)
}
