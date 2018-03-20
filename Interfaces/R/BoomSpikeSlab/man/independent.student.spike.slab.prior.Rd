\name{independent.student.spike.slab.prior}

\Rdversion{1.1}
\alias{StudentIndependentSpikeSlabPrior}
\title{
  Spike and Slab Prior for Regressions with Student T Errors
}

\description{ A spike and slab prior on the parameters of a regression
  model with Student T errors.  The prior assumes independence amon the
  regression coefficients.  }

\usage{
StudentIndependentSpikeSlabPrior(
    predictor.matrix = NULL,
    response.vector = NULL,
    expected.r2 = .5,
    prior.df = .01,
    expected.model.size = 1,
    prior.beta.sd = NULL,
    optional.coefficient.estimate = NULL,
    mean.y = mean(response.vector, na.rm = TRUE),
    sdy = sd(as.numeric(response.vector), na.rm = TRUE),
    sdx = apply(as.matrix(predictor.matrix), 2, sd, na.rm = TRUE),
    prior.inclusion.probabilities = NULL,
    number.of.observations = nrow(predictor.matrix),
    number.of.variables = ncol(predictor.matrix),
    scale.by.residual.variance = FALSE,
    sigma.upper.limit = Inf,
    degrees.of.freedom.prior = UniformPrior(.1, 100))
}

\arguments{
  \item{predictor.matrix}{
    The design matrix for the regression problem.  Missing data is not allowed.
  }

  \item{response.vector}{
    The vector of responses for the regression.  Missing data is not allowed.
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
    the number of significant predictor p variables.  Used to obtain the
    'spike' portion of the spike and slab prior.
  }

  \item{prior.beta.sd}{
   A vector of positive numbers giving the prior standard deviation of
   each model coefficient, conditionl on inclusion.  If NULL it will be
   set to 10 * the ratio of sdy / sdx.
  }

  \item{optional.coefficient.estimate}{ If desired, an estimate of the
    regression coefficients can be supplied.  In most cases this will be
    a difficult parameter to specify.  If omitted then a prior mean of
    zero will be used for all coordinates except the intercept, which
    will be set to mean(y).  }

  \item{mean.y}{The mean of the response vector, for use in cases when
    specifying the response vector is undesirable.}

  \item{sdy}{The standard deviation of the response vector, for use in
    cases when specifying the response vector is undesirable.}

  \item{sdx}{The standard deviations to use when scaling the prior sd of
    each coefficient. }

  \item{prior.inclusion.probabilities}{A vector giving the prior
    probability of inclusion for each variable.}

  \item{number.of.observations}{The number of observations in the data
    to be modeled.}

  \item{number.of.variables}{The number of potential predictor variables
    in the data to be modeled.}

  \item{scale.by.residual.variance}{If \code{TRUE} the prior variance is
    \code{sigma_sq * V}, where \code{sigma_sq} is the residual variance of the
    linear regression modeled by this prior.  Otherwise the prior
    variance is \code{V}, unscaled. }

  \item{sigma.upper.limit}{The largest acceptable value for the residual
    standard deviation.  A non-positive number is interpreted as
    \code{Inf}.
  }

  \item{degrees.of.freedom.prior}{An object of class
    \code{\link[Boom]{DoubleModel}} representing the prior distribution for the
    Student T tail thickness (or "degrees of freedom") parameter.}
}

\value{
  An \code{\link{IndependentSpikeSlabPrior}} with
  \code{degrees.of.freedom.prior} appended.
}

\author{
  Steven L. Scott
}

\references{
  Ghosh and Clyde (2011) "Rao-Blackwellization for Bayesian variable
  selection and model averaging in linear and binary regression: A novel
  data augmentation approach", \emph{Journal of the American Statistical
  Association}, \bold{106} 1041-1052.
  \url{http://homepage.stat.uiowa.edu/~jghsh/ghosh_clyde_2011_jasa.pdf}

}
