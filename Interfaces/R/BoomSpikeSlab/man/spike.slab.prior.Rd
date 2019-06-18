% Copyright 2010 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{spike.slab.prior}
\Rdversion{1.1}
\alias{SpikeSlabPrior}
\alias{SpikeSlabPriorDirect}
\alias{ConditionalZellnerPrior}

\title{
  Create a spike and slab prior for use with lm.spike.
}
\description{
  Creates a spike and slab prior for use with lm.spike.
}
\usage{
SpikeSlabPrior(x,
               y = NULL,
               expected.r2 = .5,
               prior.df = .01,
               expected.model.size = 1,
               prior.information.weight = .01,
               diagonal.shrinkage = .5,
               optional.coefficient.estimate = NULL,
               max.flips = -1,
               mean.y = mean(y, na.rm = TRUE),
               sdy = sd(as.numeric(y), na.rm = TRUE),
               prior.inclusion.probabilities = NULL,
               sigma.upper.limit = Inf)

SpikeSlabPriorDirect(coefficient.mean,
                     coefficient.precision,
                     prior.inclusion.probabilities,
                     prior.df,
                     sigma.guess,
                     max.flips = -1,
                     sigma.upper.limit = Inf)


ConditionalZellnerPrior(xdim, 
                        optional.coefficient.estimate = NULL,
                        expected.model.size = 1,
                        prior.information.weight = .01,
                        diagonal.shrinkage = .5,
                        max.flips = -1,
                        prior.inclusion.probabilities = NULL)

}
\arguments{
  \item{x}{
    The design matrix for the regression problem.  Missing data is not allowed.
  }

  \item{y}{ The vector of responses for the regression.  Missing data is
    not allowed.  If \code{y} is not available, you can pass \code{y =
      NULL}, and specify \code{mean.y} and \code{sdy} instead.
  }

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
    the sampler will attempt to sample each iteration.  If
    \code{max.flips <= 0} then all indicators will be sampled.}

  \item{mean.y}{The mean of the response vector, for use in cases when
    specifying the response vector is undesirable.}

  \item{xdim}{The dimension of the predictor matrix.}
  
  \item{sdy}{The standard deviation of the response vector, for use in
    cases when specifying the response vector is undesirable.}

  \item{prior.inclusion.probabilities}{A vector giving the prior
    probability of inclusion for each variable.}

  \item{sigma.upper.limit}{The largest acceptable value for the residual
    standard deviation.  A non-positive number is interpreted as
    \code{Inf}.}

  \item{coefficient.mean}{The prior mean of the coefficients in the
    maximal model (with all variables included).}

  \item{coefficient.precision}{The prior precision (inverse variance) of
    the coefficients in the maximal model (with all variables
    included).}

  \item{sigma.guess}{Prior estimate of the residual standard deviation.}
  
}

\value{
  A list with with the components necessary to run \code{lm.spike}.

  \code{SpikeSlabPrior} is intended for use in traditional regression
  problems, when the matrix of predictors and the vector of responses
  are available to the modeler.

  \code{ConditionalZellnerPrior} is intended for cases where the
  predictor variables are potentially unknown, because they depend on
  model parameters or latent variables, for example.  For models that
  support ConditionalZellnerPrior, the underlying C++ code must know
  where to find the relevant predictors on which to condition the prior.
  
}

\author{
  Steven L. Scott
}

\references{
  George and McCulloch (1997), "Approaches to Bayesian Variable
  Selection", \emph{Statistica Sinica}, \bold{7}, 339 -- 373.

  \url{http://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf}
}

\examples{
  x <- cbind(1, matrix(rnorm(900), ncol = 9))
  beta <- rep(0, 10)
  beta[1] <- 3
  beta[5] <- -4
  beta[8] <- 2
  y <- rnorm(100, x \%*\% beta)
  ## x has 10 columns, including the intercept
  prior <- SpikeSlabPrior(x, y,
             expected.model.size = 3,  # expect 3 nonzero predictors
             prior.df = .01,           # weaker prior than the default
             prior.information.weight = .01,
             diagonal.shrinkage = 0,   # use Zellner's prior
             optional.coefficient.estimate = rep(0, 10) # shrink to zero
          )
  ## now 'prior' can be fed to 'lm.spike'
  model <- lm.spike(y ~ x - 1, niter = 1000, prior = prior)
}
\keyword{models}
\keyword{regression}
