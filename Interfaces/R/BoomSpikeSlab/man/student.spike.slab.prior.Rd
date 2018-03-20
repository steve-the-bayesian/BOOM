\name{student.spike.slab.prior}
\Rdversion{1.1}
\alias{StudentSpikeSlabPrior}
\title{
  Spike and Slab Prior for Student-T Regression
}
\description{
  A Zellner-style spike and slab prior for regression models with
  Student-t errors.
}
\usage{
StudentSpikeSlabPrior(predictor.matrix,
    response.vector = NULL,
    expected.r2 = .5,
    prior.df = .01,
    expected.model.size = 1,
    prior.information.weight = .01,
    diagonal.shrinkage = .5,
    optional.coefficient.estimate = NULL,
    max.flips = -1,
    mean.y = mean(response.vector, na.rm = TRUE),
    sdy = sd(as.numeric(response.vector), na.rm = TRUE),
    prior.inclusion.probabilities = NULL,
    sigma.upper.limit = Inf,
    degrees.of.freedom.prior = UniformPrior(.1, 100))
}

\arguments{
  \item{predictor.matrix}{
    The design matrix for the regression problem.  Missing data is not allowed.
  }

  \item{response.vector}{ The vector of responses for the regression.
    Missing data is not allowed.  If \code{response.vector} is not
    available, you can pass \code{response.vector = NULL}, and specify
    \code{mean.y} and \code{sdy} instead.  }

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

  \item{sdy}{The standard deviation of the response vector, for use in
    cases when specifying the response vector is undesirable.}

  \item{prior.inclusion.probabilities}{A vector giving the prior
    probability of inclusion for each variable.}

  \item{sigma.upper.limit}{The largest acceptable value for the residual
    standard deviation.  A non-positive number is interpreted as
    \code{Inf}.}

  \item{degrees.of.freedom.prior}{An object of class
    \code{\link[Boom]{DoubleModel}} representing the prior distribution for the
    Student T tail thickness (or "degrees of freedom") parameter.}
}

\value{ A \code{\link{SpikeSlabPrior}} with
  \code{degrees.of.freedom.prior} appended.  } \author{ Steven L. Scott
}

\references{
  George and McCulloch (1997), "Approaches to Bayesian Variable
  Selection", \emph{Statistica Sinica}, \bold{7}, 339 -- 373.

  \url{http://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf}
}

\keyword{models}
\keyword{regression}
