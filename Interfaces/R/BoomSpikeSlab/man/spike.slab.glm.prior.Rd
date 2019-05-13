\name{spike.slab.glm.prior}

\Rdversion{1.1}
\alias{SpikeSlabGlmPrior}
\alias{SpikeSlabGlmPriorDirect}

\title{
  Zellner Prior for Glm's.
}

\description{
  A Zellner-style spike and slab prior for generalized linear models.
  It is intended as a base class for LogitZellnerPrior,
  PoissonZellnerPrior, and potential future extensions.
}

\usage{
SpikeSlabGlmPrior(
    predictors,
    weight,
    mean.on.natural.scale,
    expected.model.size,
    prior.information.weight,
    diagonal.shrinkage,
    optional.coefficient.estimate,
    max.flips,
    prior.inclusion.probabilities)

SpikeSlabGlmPriorDirect(
    coefficient.mean,
    coefficient.precision,
    prior.inclusion.probabilities = NULL,
    expected.model.size = NULL,
    max.flips = -1)

}

\arguments{
  \item{predictors}{
    The design matrix for the regression problem.  No missing data is allowed.
  }

  \item{weight}{A vector of length \code{nrow(predictors)} giving the
    prior weight assigned to each observation in \code{predictors}.
    This should ideally match the weights from the Fisher information
    (e.g. \code{p * (1-p)}) for logistic regression, or \code{lambda} for
    Poisson regression, but that depends on the model, so a typical
    thing to do is to set all the weights the same. }

  \item{mean.on.natural.scale}{Used to set the prior mean for the
    intercept.  The mean of the response, expressed on the natural
    scale.  This is logit(p-hat) for logits and log(ybar) for Poissons.
    }

  \item{expected.model.size}{
    A positive number less than \code{ncol(x)}, representing a guess at
    the number of significant predictor variables.  Used to obtain the
    'spike' portion of the spike and slab prior.
  }

  \item{prior.information.weight}{
    A positive scalar.  Number of observations worth of weight that
    should be given to the prior estimate of beta.
  }

  \item{diagonal.shrinkage}{
    The conditionally Gaussian prior for beta (the "slab") starts with a
    precision matrix equal to the information in a single observation.
    However, this matrix might not be full rank.  The matrix can be made
    full rank by averaging with its diagonal.  \code{diagonal.shrinkage}
    is the weight given to the diaonal in this average.  Setting this to
    zero gives Zellner's g-prior.
  }

  \item{optional.coefficient.estimate}{ If desired, an estimate of the
    regression coefficients can be supplied.  In most cases this will be
    a difficult parameter to specify.  If omitted then a prior mean of
    zero will be used for all coordinates except the intercept, which
    will be set to mean(y).  }

  \item{max.flips}{The maximum number of variable inclusion indicators
    the sampler will attempt to sample each iteration.  If negative then
    all indicators will be sampled.}

  \item{prior.inclusion.probabilities}{A vector giving the prior
    probability of inclusion for each variable.  If \code{NULL} then a
    default set of probabilities is obtained by setting each element
    equal to \code{min(1, expected.model.size / ncol(x))}.}

  \item{coefficient.mean}{The prior mean of the coefficients in the
    maximal model (with all coefficients included).}

  \item{coefficient.precision}{The prior precision (inverse variance) of
    the coefficients in the maximal model (with all coefficients
    included).}
  
}

\details{ A Zellner-style spike and slab prior for generalized linear
  models.  Denote the vector of coefficients by \eqn{\beta}{beta}, and the
  vector of inclusion indicators by \eqn{\gamma}{gamma}.  These are linked
  by the relationship \eqn{\beta_i \ne 0}{beta[i] != 0} if \eqn{\gamma_i =
    1}{gamma[i] = 1} and \eqn{\beta_i = 0}{beta[i] == 0} if \eqn{\gamma_i =
    0}{gamma[i] = 0}.  The prior is

\deqn{\beta | \gamma \sim N(b, V)}{beta | gamma ~ N(b, V),}
\deqn{\gamma \sim B(\pi)}{gamma ~ Bernoulli(pi)}

where \eqn{\pi}{pi} is the vector of
\code{prior.inclusion.probabilities}, and \eqn{b} is the
\code{optional.coefficient.estimate}.  Conditional on
\eqn{\gamma}{gamma}, the prior information matrix is

\deqn{V^{-1} = \kappa ((1 - \alpha) x^Twx / n + \alpha diag(x^Twx / n))}{
  V^{-1} = kappa * ((1 - alpha) * x^Twx / n + alpha * diag(x^Twx/n))
}

The matrix \eqn{x^Twx} is, for suitable choice of the weight vector
\eqn{w}, the total Fisher information available in the data.
Dividing by \eqn{n} gives the average Fisher information in a single
observation, multiplying by \eqn{\kappa}{kappa} then results in
\eqn{\kappa}{kappa} units of "average" information.  This matrix is
averaged with its diagonal to ensure positive definiteness.

In the formula above, \eqn{\kappa}{kappa} is
\code{prior.information.weight}, \eqn{\alpha}{alpha} is
\code{diagonal.shrinkage}, and \eqn{w} is a diagonal matrix with all
elements set to \code{prior.success.probability * (1 -
  prior.success.probability)}.  The vector \eqn{b} and the matrix
\eqn{V^{-1}} are both implicitly subscripted by \eqn{\gamma}{gamma},
meaning that elements, rows, or columsn corresponding to gamma = 0
should be omitted.

The "Direct" version is intended for situations where the predictors are
unavailable, or if the user wants more control over the prior precision
matrix.

}

\value{

  Returns an object of class \code{SpikeSlabGlmPrior}, which is a list
  with data elements encoding the selected prior values.

  This object is intended for use as a base class for
  \code{\link{LogitZellnerPrior}} and \code{\link{PoissonZellnerPrior}}.
}

\author{
  Steven L. Scott
}

\references{
  Hugh Chipman, Edward I. George, Robert E. McCulloch, M. Clyde, Dean
  P. Foster, Robert A. Stine (2001),
    "The Practical Implementation of Bayesian Model Selection"
    \emph{Lecture Notes-Monograph Series}, Vol. 38, pp. 65-134.
  Institute of Mathematical Statistics.
}
