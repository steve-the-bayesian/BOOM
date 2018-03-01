% Copyright 2016 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{qreg.spike}
\Rdversion{1.1}
\alias{qreg.spike}
\title{
  Quantile Regression
}
\description{
  MCMC algorithm for quasi-Bayesian quantile models with a 'spike-and-slab'
  prior that places some amount of posterior probability at zero for a
  subset of the regression coefficients.
}
\usage{
qreg.spike(formula,
               quantile,
               niter,
               ping = niter / 10,
               nthreads = 0,
               data,
               subset,
               prior = NULL,
               na.action = options("na.action"),
               contrasts = NULL,
               drop.unused.levels = TRUE,
               initial.value = NULL,
               seed = NULL,
               ...)
}
\arguments{

  \item{formula}{Formula for the maximal model (with all variables
    included).  }

  \item{quantile}{A scalar value between 0 and 1 indicating the
    quantile of the conditional distribution being modeled.}

  \item{niter}{ The number of MCMC iterations to run.  Be sure to
    include enough so you can throw away a burn-in set.  }

  \item{ping}{If positive, then print a status update to the console
    every \code{ping} MCMC iterations.}

  \item{nthreads}{The number of CPU-threads to use for data
    augmentation.  There is some small overhead to stopping and starting
    threads.  For small data sets, thread overhead will make it faster
    to run single threaded.  For larger data sets multi-threading can
    speed things up substantially.  This is all machine dependent, so
    please experiment.}

  \item{data}{ An optional data frame, list or environment (or object
    coercible by \code{as.data.frame} to a data frame) containing the
    variables in the model.  If not found in \code{data}, the variables
    are taken from \code{environment(formula)}, typically the
    environment from which \code{qreg.spike} is called.  }

  \item{subset}{ An optional vector specifying a subset of observations
    to be used in the fitting process.  }

  \item{prior}{An optional list such as that returned from
    \code{\link{SpikeSlabPrior}}.  If missing, \code{SpikeSlabPrior}
    will be called using the extra arguments passed via \dots.
  }

  \item{na.action}{A function which indicates what should happen when
    the data contain \code{NA}s.  The default is set by the
    \code{na.action} setting of \code{options}, and is \code{na.fail} if
    that is unset.  The \code{factory-fresh} default is \code{na.omit}.
    Another possible value is \code{NULL}, no action.  Value
    \code{na.exclude} can be useful.}

  \item{contrasts}{ An optional list. See the \code{contrasts.arg} of
    \code{\link{model.matrix.default}}.  }

  \item{drop.unused.levels}{ A logical value indicating whether factor
    levels that are unobserved should be dropped from the model.}

  \item{initial.value}{Initial value for the MCMC algorithm.  Can either
    be a numeric vector, a \code{\link{glm}} object (from which the
    coefficients will be used), or a \code{\link{qreg.spike}} object.
    If a \code{\link{qreg.spike}} object is supplied, it is assumed to
    be from a previous MCMC run for which \code{niter} additional draws
    are desired.  If a \code{\link{glm}} object is supplied then its
    coefficients will be used as the initial values for the simulation.
  }

  \item{seed}{Seed to use for the C++ random number generator.  It
    should be \code{NULL} or an int.  If \code{NULL} the seed value will
    be taken from the global \code{\link{.Random.seed}} object.}

  \item{\dots}{
    Extra arguments to be passed to \code{\link{SpikeSlabPrior}}.
  }

}
\value{

  Returns an object of class \code{qreg.spike}, which inherits from
  \code{lm.spike}.  The returned object is a list with the following
  elements

  \item{beta}{A \code{niter} by \code{ncol(x)} matrix of regression
    coefficients, many of which may be zero.  Each row corresponds to an
    MCMC iteration.}

  \item{prior}{The prior used to fit the model.  If a \code{prior} was
    supplied as an argument it will be returned.  Otherwise this will be
    the automatically generated prior based on the other function
    arguments. }

}

\details{

  Just like ordinary regression models the mean of a distribution as a
  linear function of X, quantile regression models a specific quantile
  (e.g. the 90th percentile) as a function of X.

  Median regression is a special case of quantile regression.  Median
  regression is sometimes cast in terms of minimizing |y - X * beta|,
  because the median is the optimal action under L1 loss.  Similarly,
  selecting quantile tau is optimal under the asymmetric loss function

  \deqn{\rho_\tau(u) = \tau u I(u > 0) + (1-\tau) u I(u < 0) }{%
    rho_tau(u) = tau * u * I(u > 0) + (1 - tau) * u * I(u < 0)}

  Thus quantile regression (for a specific quantile tau) minimizes
  \deqn{ Q(\beta) = \sum_i \rho_\tau( y_i - \beta^Tx_i) }{%
    Q(beta) = sum(rho_tau(y - X \%*\% beta))}

  Bayesian quantile regression treats \deqn{\exp(-2Q(\beta))}{-2Q(beta)}
  as a likelihood function to which a prior distribution
  \eqn{p(\beta)}{p(beta)} is applied.  For posterior sampling, a data
  augmentation scheme is used where each observation is associated with
  a latent variable \eqn{\lambda_i}{lambda}, which has a marginal
  distribution of \deqn{Exp(2 \tau(1-\tau)).}{Exp(2 * tau * (1 - tau)).}

  The conditional distribution given the residual \eqn{r = y - x
    \beta}{r = y - x * beta} is
  \deqn{
    \frac{1}{\lambda} | r \sim  InvGaus( 1 / |r|, 1.0)}{%
    (1 / lambda) | r  ~ InverseGaussian(1 / |r|, 1.0)}

  The conditional distribution of beta given complete data (lambda and
  y) is a weighted least squares regression, where observation i has
  precision \eqn{\lambda_i}{lambda[i]} and where observation i is offset
  by \eqn{2(\tau - 1)\lambda_i}{2 * (tau - 1) * lambda[i]}.

}


\references{
  Parzen and Polson (2011, unpublished)
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}},
  \code{\link{plot.qreg.spike}},
  \code{\link{predict.qreg.spike}}.
}

\examples{
  n <- 50
  x <- rnorm(n)
  y <- rnorm(n, 4 * x)
  model <- qreg.spike(y ~ x,
                      quantile = .8,
                      niter = 1000,
                      expected.model.size = 100)

  ## Should get a slope near 4 and an intercept near qnorm(.8).
  PlotManyTs(model$beta[-(1:100),],
             same.scale = TRUE,
             truth = c(qnorm(.8), 4))

}
\keyword{models}
\keyword{regression}
