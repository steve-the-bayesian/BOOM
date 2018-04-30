\name{BoomBart}

\alias{BoomBart}
\alias{Bart}
\alias{bart}

\title{Bayesian additive regression trees}
\Rdversion{1.0}
\description{
  Bayesian additive regression tress, with Gaussian, probit, logit, or
  Poisson errors.
}

\usage{
BoomBart(formula, niter, data,
         family = c("gaussian", "probit", "logit", "poisson"),
         initial.number.of.trees = 200,
         tree.prior = NULL,
         discrete.distribution.limit = 20,
         continuous.distribution.strategy = c(
           "uniform.continuous",
           "uniform.discrete"),
         ping = niter / 10,
         seed = NULL,
         total.prediction.sd = NULL,
         number.of.trees.prior = DiscreteUniformPrior(1, 200),
         ...)
}

\arguments{
  \item{formula}{A model formula such as you might give to
    \code{\link{lm}}, \code{\link{glm}}, etc.  The formula should only
    consist of linear terms as the model will figure out any
    interactions or nonlinearities on its own.}

  \item{niter}{The number of MCMC iterations to run.}

  \item{data}{An optional data.frame containing the variables used in
    \code{formula}.}

  \item{family}{A string indicating the family (and link function) of
    the error distribution.}

  \item{initial.number.of.trees}{The number of trees used to initialize
    the model.  This number must have positive probability under
    \code{number.of.trees.prior}.}

  \item{tree.prior}{An object inheriting from
    \code{\link{BartTreePrior}} specifying the prior over tree topology,
    the number of trees, the mean parameters at the leaves, and (for
    Gaussian models) the residual variance.  If \code{NULL} a default
    prior will be generated, taking the \code{total.prediction.sd},
    \code{number.of.trees.prior}, and \code{...} arguments into
    consideration.  }

  \item{discrete.distribution.limit}{The number of unique values below
    which a numeric variable will be considered "discrete".  A discrete
    variable has potential cutpoints at its unique values.}

  \item{continuous.distribution.strategy}{ The strategy to use when
    creating cutpoints for continuous variables.  The available
    strategies are:
    \itemize{

      \item{uniform.continuous: cutpoints are chosen at random from the
	continuous range between the largest and smallest observed
	values of a variable.}

      \item{uniform.discrete: cutpoints are chosen from a discretization
	of the variable.  The number of cutpoints is the smaller of 100,
	or the number of unique values in the variable.}

    }
  }

  \item{ping}{The frequency with which to print status update messages
    during the MCMC.  For example, \code{ping = 10} will print a message
    every 10 iterations.}

  \item{seed}{The random seed to use for the BOOM random number
    generator.  An integer.}

  \item{total.prediction.sd}{The prior standard deviation of the
    predicted values.  This value is only used if \code{tree.prior} is
    \code{NULL}.  If this is \code{NULL} as well, it will be set to a
    value that depends on the \code{family} argument.  For Gaussia
    models the default value is \code{sd(y)}.  For probit families the
    default value is 1.0, for logit it is 1.8, and for Poisson the
    default is \code{.5 * log(mean(y))}.

    The prior distribution for the leaf mean parameters is normal with
    standard deviation \code{total.prediction.sd / sqrt(number.of.trees)}.
  }

  \item{number.of.trees.prior}{A prior distribution of class
    \code{\link[Boom]{DiscreteUniformPrior}}.  See
    \code{?DiscreteUniformPrior} in the \code{Boom} package.}

  \item{...}{Extra arguments are passed either to
    \code{\link{GaussianBartTreePrior}} (for Gaussian models) or
    \code{\link{BartTreePrior}} (for Poisson, logit or probit models). }

}

\value{ An object of class BoomBart, which is a list containing Monte
  Carlo draws of the trees (represented as lists of matrices), and (if
  \code{family == "gaussian"}) the residual standard deviation "sigma".
  Most of the output is not designed for users to examine directly, but
  rather to be used in plot, predict, and related methods.


  The MCMC draws of the Bart trees are also included in the returned
  object.  These are stored as a list of matrices, where each matrix
  represents one MCMC draw of the Bart ensemble.  Each row in the matrix
  represents a node in a tree.  The four columns of the matrix represent
  \enumerate{
    \item The id of the parent (with the root having parent id 0),

    \item The index of the node's decision variable in the design
    matrix, which is -1 if the node is a leaf.

    \item If the node is a leaf, the third column contains the node's
    mean parameter.  Otherwise, the third column contains the cutpoint
    for the node's decision variable.
  }

}

\references{
Chipman, George, McCulloch (2010) BART: Bayesian additive regression
trees.  Annals of Applied Statistics.  Volume 4, Number 1. 266 -- 298.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
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

\keyword{models}
\keyword{regression}
