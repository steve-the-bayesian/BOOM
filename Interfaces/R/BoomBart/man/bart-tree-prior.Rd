\name{bart-tree-prior}
\alias{BartTreePrior}
\alias{GaussianBartTreePrior}

\title{Bart tree prior}

\Rdversion{1.0}
\description{Prior distributions for Bart models.}

\usage{
BartTreePrior(total.prediction.sd,
              root.split.probability = .95,
              split.decay.rate = 2,
              number.of.trees.prior =
                DiscreteUniformPrior(1, 200))

GaussianBartTreePrior(
    total.prediction.sd,
    root.split.probability = .95,
    split.decay.rate = 2,
    number.of.trees.prior = DiscreteUniformPrior(1, 200),
    sigma.guess = sqrt(sdy^2 * (1 - expected.r2)),
    sigma.weight = 3,
    expected.r2 = .5,
    sdy)
}

\arguments{

  \item{total.prediction.sd}{
    The amount by which you expect the predictions to vary, on the "link
    scale" (i.e. the raw scale for Gaussian models, the probit scale for
    probits, the logit scale for logits, and the log scale for Poisson).
    This parameter determines the prior for the mean parameters at the
    leaves of the trees, which is N(0, total.prediction.sd^2 /
    number.of.trees)
  }

  \item{root.split.probability}{The probability that a split occurs at
    the root.  This should be set close to 1.}

  \item{split.decay.rate}{The exponent that determines how quickly the
    probability of a node splitting diminishes as a function of depth.
    See the 'details' section.}

  \item{number.of.trees.prior}{A prior distribution of class
    \code{\link[Boom]{DiscreteUniformPrior}} giving the prior
    distribution on the number of trees.  See
    \code{?DiscreteUniformPrior} in the \code{Boom} package.}

  \item{sigma.guess}{For Gaussian models only.  A guess at the residual
    standard deviation.  By default this is determined empirically as a
    fraction of the standard deviation of the data, but it can be
    supplied directly if desired. }

  \item{sigma.weight}{For Gaussian models only.  The the number of
  observations worth of weight given to \code{sigma.guess}.}

  \item{expected.r2}{For Gaussian models only.  The "expected R^2" from
    the regression.  This is only used to set \code{sigma.guess}.}

  \item{sdy}{For Gaussian models only.  The raw standard deviation of
    the respsonse variable.  This is only used to set \code{sigma.guess}.
  }
}

\value{Returns an object of class \code{BartTreePrior}, which is a list
  with the following elements

  \itemize{
    \item{total.prediction.sd}
    \item{root.split.probability}
    \item{split.decay.rate}
    \item{number.of.trees.prior}
  }

  A \code{GaussianBartTreePrior} contains an additional element
  \code{sd.prior}, which is an object of class
  \code{\link[Boom]{SdPrior}} giving the prior distribution
  on the residual variance. }

\details{
  The prior on a non-Gaussian Bart model contains three components, a
  prior on the tree topology, a prior on the number of trees, and a
  prior on the values of the mean parameters at the leaf nodes.  A
  fourth prior component over the decision variables and cutpoints a the
  interior nodes, is present but not specified because it is always
  uniform.

  The prior over tree topology is determined by the probability of a
  split at each node.  The depth of a node is the minimal number of
  steps to the root.  So the root is at depth 0, its children at depth
  1, etc.  The probability that the root has children is \eqn{alpha}.
  The probability that a node at depth \eqn{d} has children is \deqn{
    alpha / (1 + d)^beta.}

  As a function argument \eqn{\alpha} is called
  \code{root.split.probability} and \eqn{\beta} is called
  \code{split.decay.rate}.

  The prior on the mean parameters at the leaves is
  \deqn{
    N(0, \tau^2 / F)
  }
  where \eqn{F} is the number of trees and \eqn{\tau} is the
  \code{total.prediction.sd} function argument.

  It is possible to fit a model with a fixed number of trees by setting
  \code{number.of.trees.prior = PointMassPrior(n)}, where \code{n} is
  the desired number.
}

\references{
Chipman, George, McCulloch (2010) BART: Bayesian additive regression
trees.  Annals of Applied Statistics.  Volume 4, Number 1. 266 -- 298.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
library(Boom)
data(BostonHousing)
niter <- 500  ## Make larger in a real application
cv <- sample(1:nrow(BostonHousing), size = 50, replace = FALSE)
train <- (1:nrow(BostonHousing))[-cv]
model <- BoomBart(medv ~ ., data = BostonHousing[train, ], niter = niter)
pred <- predict(model, newdata = BostonHousing[cv, ], burn = 250)
BoxplotTrue(pred,
            truth = BostonHousing[cv, ]$medv,
            center = TRUE,
            ylab = "Prediction Error")

}
