% Copyright 2010 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{lm.spike}
\Rdversion{1.1}
\alias{lm.spike}
\alias{spikeslab}
\alias{SsvsOptions}
\alias{OdaOptions}
\title{
  Spike and slab regression
}
\description{
  MCMC algorithm for linear regression models with a 'spike-and-slab'
  prior that places some amount of posterior probability at zero for a
  subset of the regression coefficients.

  The model admits either Gaussian or student T errors; the latter are
  useful in the presence of outliers.
}
\usage{
lm.spike(formula,
         niter,
         data,
         subset,
         prior = NULL,
         error.distribution = c("gaussian", "student"),
         contrasts = NULL,
         drop.unused.levels = TRUE,
         model.options = SsvsOptions(),
         ping = niter / 10,
         seed = NULL,
         ...)

SsvsOptions(adaptive.cutoff = 100,
            adaptive.step.size = .001,
            target.acceptance.rate = .345,
            correlation.swap.threshold = .8)

OdaOptions(fallback.probability = 0.0,
           eigenvalue.fudge.factor = 0.01)
}

\arguments{
  \item{formula}{
    formula for the maximal model (with all variables included), this is
    parsed the same way as a call to \code{lm}.
  }
  \item{niter}{
    The number of MCMC iterations to run.  Be sure to include enough so
    you can throw away a burn-in set.
  }
  \item{data}{
    An optional data frame, list or environment (or object coercible by
    'as.data.frame' to a data frame) containing the variables in the
    model.  If not found in 'data', the variables are taken from
    'environment(formula)', typically the environment from which
    'lm.spike' is called.
  }
  \item{subset}{
    An optional vector specifying a subset of observations to be used in
    the fitting process.
  }

  \item{prior}{ An optional list returned by
    \code{\link{SpikeSlabPrior}}.  If \code{prior} is missing then a
    default prior will be used.  See \code{\link{SpikeSlabPrior}}.
  }

  \item{error.distribution}{Specify either Gaussian or Student T
    errors.  If the error distribution is student then the prior
    must be a \code{\link{StudentSpikeSlabPrior}} and the SSVS method
    must be used.}

  \item{contrasts}{ An optional list. See the \code{contrasts.arg}
    argument of \code{\link{model.matrix.default}}.  }

  \item{drop.unused.levels}{Logical indicating whether unobserved factor
    levels should be dropped from the model.}

  \item{model.options}{A list containing the tuning parameters for the
    desired MCMC method.  }
  
  \item{ping}{The frequency with which to print status update messages
    to the screen.  For example, if \code{ping == 10} then an update
    will be printed every 10 MCMC iterations.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{\dots}{
    Extra arguments to be passed to \code{\link{SpikeSlabPrior}} (if
    \code{method == "SSVS"}) or \code{\link{IndependentSpikeSlabPrior}}
    (if \code{method == "ODA"}).}

  \item{fallback.probability}{ When using the ODA method, each MCMC
    iteration will use SSVS instead of ODA with this probability.  In
    cases where the latent data have high leverage, ODA mixing can
    suffer.  Mixing in a few SSVS steps can help keep an errant
    algorithm on track.}

  \item{eigenvalue.fudge.factor}{When using the ODA method, the latent
    X's will be chosen so that the complete data X'X matrix (after
    scaling) is a constant diagonal matrix equal to the largest
    eigenvalue of the observed (scaled) X'X times (1 +
    eigenvalue.fudge.factor).  This should be a small positive number.
  }

  \item{adaptive.cutoff}{The traditional SSVS method (sample every
    predictor at every iteration) will be used when there are fewer than
    this many predictors.  The adaptive method of Benson and Fried will
    be used if there are more.}

  \item{adaptive.step.size}{The step size scaling factor to use in the
    adaptive SSVS algorithm.}

  \item{target.acceptance.rate}{The target acceptance rate for the
    adaptive SSVS algorithm.}

  \item{correlation.swap.threshold}{ The minimal absolute correlation
    required for two variables to be considered for a swap move.  Swap
    moves are currently only supported for less than
    \code{adaptive.cutoff} variables.  }

}

\details{

  There are two MCMC methods available.  SSVS is the stochastic search
  variable selection algorithm from George and McCulloch (1998).  ODA is
  the orthogonal data augmentation method from Clyde and Ghosh (2011).
  Both sampling methods ("ODA" and "SSVS") draw each variable inclusion
  indicator given all others, in a Gibbs sampler.  The ODA method
  includes an extra data augmentation step that renders each indicator
  conditionally independent of the others given the latent data.  There
  is residual dependence between successive MCMC steps introduced by the
  latent data, but the paper by Ghosh and Clyde suggested that on
  balance mixing should be improved.

  SSVS offers a choice between to implementations.  Classic SSVS
  attempts to flip each coefficient in or out of the model every
  iteration.  The adaptive method attempts to learn which coefficients
  are likely to be included or excluded.  It then biases its 'birth' and
  'death' moves towards candidates that are likely to succeed.
  
  Regarding the overall compute time, the DA method decomposes the
  (potentially very large) model matrix one time, at the start of the
  algorithm.  But it then works with independent scalar updates.  The
  SSVS algorithm does not have the upfront cost, but it works with many
  small matrix decompositions each MCMC iteration.  The DA algorithm is
  very likely to be faster in terms of time per iteration.

  Finally, note that the two algorithms require slightly different
  priors.  The DA algorithm requires a priori independence, while the
  SSVS algorithm can work with arbitrary conjugate priors.
}

\value{
  Returns an object of class \code{lm.spike}, which is a list with the
  following elements
  \item{beta}{A \code{niter} by \code{ncol(x)} matrix of regression
    coefficients, many of which may be zero.  Each row corresponds to an
    MCMC iteration. }

  \item{sigma}{ A vector of length \code{niter} containing the MCMC
    draws of the residual standard deviation parameter. }

  \item{prior}{The prior used to fit the model.  If a \code{prior} was
    supplied as an argument it will be returned.  Otherwise this will be
    the automatically generated prior based on the other function
    arguments. }

}

\references{
  George and McCulloch (1997), "Approaches to Bayesian Variable
  Selection", \emph{Statistica Sinica}, \bold{7}, 339 -- 373.
  \url{https://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf}

  Ghosh and Clyde (2011) "Rao-Blackwellization for Bayesian variable
  selection and model averaging in linear and binary regression: A novel
  data augmentation approach", \emph{Journal of the American Statistical
  Association}, \bold{106} 1041-1052.
  \url{https://homepage.stat.uiowa.edu/~jghsh/ghosh_clyde_2011_jasa.pdf}
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{SpikeSlabPrior}},
  \code{\link{plot.lm.spike}},
  \code{\link{summary.lm.spike}},
  \code{\link{predict.lm.spike}}.
}

\examples{
  n <- 100
  p <- 10
  ngood <- 3
  niter <- 1000
  sigma <- .8

  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  y <- rnorm(n, x \%*\% beta, sigma)
  x <- x[,-1]
  model <- lm.spike(y ~ x, niter=niter)
  plot.ts(model$beta)
  hist(model$sigma)  ## should be near 8
  plot(model)
  summary(model)
  plot(model, "residuals")

  ## Now replace the first observation with a big outlier.
  y[1] <- 50
  model <- lm.spike(y ~ x, niter = niter)
  model2 <- lm.spike(y ~ x, niter = niter, error.distribution = "student")
  pred <- predict(model, newdata = x)
  pred2 <- predict(model2, newdata = x)

  ## Maximize the plot window before making these box plots.  They show
  ## the posterior predictive distribution of all 100 data points, so
  ## make sure your screen is 100 boxes wide!
  par(mfrow = c(2,1))
  BoxplotTrue(t(pred), truth = y, ylim = range(pred), pch = ".",
     main = "Posterior predictive distribution assuming Gaussian errors.")
  BoxplotTrue(t(pred2), truth = y, ylim  = range(pred), pch = ",",
     main = "Posterior predictive distribution assuming Student errors.")

  ## The posterior predictive distributions are much tighter in the
  ## student case than in the Gaussian case, even though the student
  ## model has heavier tails, because the "sigma" parameter is smaller.
  par(mfrow = c(1,1))
  CompareDensities(list(gaussian = model$sigma, student = model2$sigma),
                        xlab = "sigma")
}
\keyword{models}
\keyword{regression}
