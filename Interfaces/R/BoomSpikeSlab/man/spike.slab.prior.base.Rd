\name{spike.slab.prior.base}
\Rdversion{1.1}
\alias{SpikeSlabPriorBase}
\title{Base class for spike and slab priors}

\description{
  A base class for SpikeSlabPrior and SpikeSlabPriorBase to ensure that
  elements common to both classes are handled consistently.  Users
  will not normally interact with this function.
}

\usage{
SpikeSlabPriorBase(number.of.variables,
                   expected.r2 = .5,
                   prior.df = .01,
                   expected.model.size = 1,
                   optional.coefficient.estimate = NULL,
                   mean.y,
                   sdy,
                   prior.inclusion.probabilities = NULL,
                   sigma.upper.limit = Inf)
}

\arguments{
  \item{number.of.variables}{The number of columns in \code{x}.}

  \item{expected.r2}{
    The expected R-square for the regression.  The spike and slab prior
    requires an inverse gamma prior on the residual variance of the
    regression.  The prior can be parameterized in terms of a guess at
    the residual variance, and a "degrees of freedom" representing the
    number of observations that the guess should weigh.  The guess at
    sigma^2 is set to \code{ (1-expected.r2) * var(y) }.
  }

  \item{prior.df}{
    A positive scalar representing the prior 'degrees of freedom' for
    estimating the residual variance.  This can be thought of as the
    amount of weight (expressed as an observation count) given to the
    \code{expected.r2} argument.
  }

  \item{expected.model.size}{ A positive number less than
    \code{ncol(x)}, representing a guess at the number of significant
    predictor p variables.  Used to compute a default value of
    \code{prior.inclusion.probabilities} if the latter is \code{NULL}.
  }

  \item{optional.coefficient.estimate}{ If desired, an estimate of the
    regression coefficients can be supplied.  In most cases this will be
    a difficult parameter to specify.  If omitted then a prior mean of
    zero will be used for all coordinates except the intercept, which
    will be set to \code{mean.y}.  }

  \item{mean.y}{The mean of the response vector.  Used to create a
    default value of \code{optional.coefficient.estimate} when the latter
    is \code{NULL}.}

  \item{sdy}{The standard deviation of the response vector.  Used along
    with \code{expected.r2} to create a prior estimate of the residual
    variance.}

  \item{prior.inclusion.probabilities}{A vector giving the prior
    probability of inclusion for each coefficient.}

  \item{sigma.upper.limit}{The largest acceptable value for the residual
    standard deviation.  A non-positive number is interpreted as
    \code{Inf}.}  }

\value{Returns an object of class \code{SpikeSlabPriorBase}, which is a
  list with the following elements.

  \itemize{
    \item prior.inclusion.probabilities: A vector giving the prior
    probability of inclusion for each coefficient.

    \item mu: A vector giving the prior mean of each coefficient
    conditional on inclusion.

    \item sigma.guess: A prior estimate of the residual standard
    deviation.

    \item prior.df: The number of observations worth of weight to be
    given to \code{sigma.guess}.
  }
}

\references{
  George and McCulloch (1997), "Approaches to Bayesian Variable
  Selection", \emph{Statistica Sinica}, \bold{7}, 339 -- 373.

  \url{https://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf}
}

\author{
  Steven L. Scott
}
