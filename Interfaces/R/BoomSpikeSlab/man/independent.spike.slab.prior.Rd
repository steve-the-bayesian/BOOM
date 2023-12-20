\name{independent.spike.slab.prior}

\Rdversion{1.1}
\alias{IndependentSpikeSlabPrior}
\title{
  A spike and slab prior assuming a priori independence.
}

\description{
  A spike and slab prior on the regression coefficients.  The prior
  distribution assumes coefficients to be independent.
}

\usage{
IndependentSpikeSlabPrior(x = NULL,
                          y = NULL,
                          expected.r2 = .5,
                          prior.df = .01,
                          expected.model.size = 1,
                          prior.beta.sd = NULL,
                          optional.coefficient.estimate = NULL,
                          mean.y = mean(y, na.rm = TRUE),
                          sdy = sd(as.numeric(y), na.rm = TRUE),
                          sdx = apply(as.matrix(x), 2, sd, na.rm = TRUE),
                          prior.inclusion.probabilities = NULL,
                          number.of.observations = nrow(x),
                          number.of.variables = ncol(x),
                          scale.by.residual.variance = FALSE,
                          sigma.upper.limit = Inf)
}

\arguments{
  \item{x}{
    The design matrix for the regression problem.  Missing data is not allowed.
  }

  \item{y}{
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

}

\value{
  A list with with the components necessary to run \code{lm.spike} with
  method "DA".
}

\author{
  Steven L. Scott
}

\references{
  Ghosh and Clyde (2011) "Rao-Blackwellization for Bayesian variable
  selection and model averaging in linear and binary regression: A novel
  data augmentation approach", \emph{Journal of the American Statistical
  Association}, \bold{106} 1041-1052.
  \url{https://homepage.stat.uiowa.edu/~jghsh/ghosh_clyde_2011_jasa.pdf}

}

\examples{

  x <- cbind(1, matrix(rnorm(900), ncol = 9))
  beta <- rep(0, 10)
  beta[1] <- 3
  beta[5] <- -4
  beta[8] <- 2
  y <- rnorm(100, x \%*\% beta)
  ## x has 10 columns, including the intercept
  prior <- IndependentSpikeSlabPrior(x, y,
             expected.model.size = 3,  # expect 3 nonzero predictors
             prior.df = .01,           # weaker prior than the default
             optional.coefficient.estimate = rep(0, 10) # shrink to zero
          )
  ## now 'prior' can be fed to 'lm.spike'
  x <- x[, -1]
  model <- lm.spike(y ~ x, niter = 1000, prior = prior, model.options = OdaOptions())
}
