\name{suggest-bart-burn}
\title{Suggest burn-in}
\description{Suggests a number of burn-in iterations for a Bart model.}
\alias{SuggestBartBurn}

\usage{
SuggestBartBurn(model, proportion = .1)
}

\arguments{
  \item{model}{An object of class \code{\link{BoomBart}}.}
  \item{proportion}{The proportion of MCMC iterations to remove.}
}

\value{Returns a number of iterations to remove as burn in.  This is
  simply \code{proportion} times the number of MCMC iterations in
  \code{model}.  This function is mainly for readability.}

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
burn <- SuggestBartBurn(model, .1)
## burn == 100
}
