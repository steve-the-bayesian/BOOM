\name{poisson.spike}
\Rdversion{1.1}
\alias{poisson.spike}
\title{
  Spike and slab Poisson regression
}
\description{
  MCMC algorithm for Poisson regression models with a 'spike-and-slab'
  prior that places some amount of posterior probability at zero for a
  subset of the coefficients.
}
\usage{
poisson.spike(formula,
              exposure = 1,
              niter,
              data,
              subset,
              prior = NULL,
              na.action = options("na.action"),
              contrasts = NULL,
              drop.unused.levels = TRUE,
              initial.value = NULL,
              ping = niter / 10,
              nthreads = 4,
              seed = NULL,
              ...)
}
\arguments{

  \item{formula}{A model formula, as would be passed to \code{glm},
    specifying the maximal model (i.e. the model with all predictors
    included).  }

  \item{exposure}{A vector of exposure durations matching the length of
    the response vector.  If \code{exposure} is of length 1 it will be
    recycled. }

  \item{niter}{ The number of MCMC iterations to run. }

  \item{data}{ An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing
    the variables in the model.  If not found in \code{data}, the
    variables are taken from \code{environment(formula)}, typically the
    environment from which \code{poisson.spike} is called.  }

  \item{subset}{ An optional vector specifying a subset of observations
    to be used in the fitting process.  }

  \item{prior}{ A list such as that returned by
    \code{\link{SpikeSlabPrior}}.  If \code{prior} is supplied it
    will be used.  Otherwise a prior distribution will be built using
    the remaining arguments.  See \code{\link{SpikeSlabPrior}}.
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
    coefficients will be used), or a \code{\link{poisson.spike}} object.
    If a \code{\link{poisson.spike}} object is supplied, it is assumed to
    be from a previous MCMC run for which \code{niter} additional draws
    are desired.  If a \code{\link{glm}} object is supplied then its
    coefficients will be used as the initial values for the simulation.
    }

  \item{ping}{If positive, then print a status update to the console
    every \code{ping} MCMC iterations.}

  \item{nthreads}{The number of CPU-threads to use for data
    augmentation.}

  \item{seed}{Seed to use for the C++ random number generator.  It
    should be \code{NULL} or an int.  If \code{NULL} the seed value will
    be taken from the global \code{\link{.Random.seed}} object.}

  \item{\dots}{
    Extra arguments to be passed to \code{\link{SpikeSlabPrior}}.
  }

}
\value{

  Returns an object of class \code{poisson.spike}.  The returned object
  is a list with the following elements.

  \item{beta}{A \code{niter} by \code{ncol(x)} matrix of regression
    coefficients, many of which may be zero.  Each row corresponds to an
    MCMC iteration.}

  \item{prior}{The prior used to fit the model.  If a \code{prior} was
    supplied as an argument it will be returned.  Otherwise this will be
    the automatically generated prior based on the other function
    arguments. }
}

\details{
  The MCMC algorithm used here is based on the auxiliary mixture
  sampling algorithm published by Fruhwirth-Schnatter, Fruhwirth, Held,
  and Rue (2009).
}

\references{
  Sylvia Fruhwirth-Schnatter, Rudolf Fruhwirth, Leonhard Held, and Havard Rue.
  Statistics and Computing, Volume 19 Issue 4, Pages 479-492.  December 2009
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}},
  \code{\link{plot.lm.spike}},
  \code{\link{summary.lm.spike}},
  \code{\link{predict.lm.spike}}.
}

\examples{
simulate.poisson.spike <- function(n = 100, p = 10, ngood = 3, niter=1000){
  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  lambda <- exp(x \%*\% beta)
  y <- rpois(n, lambda)
  x <- x[,-1]
  model <- poisson.spike(y ~ x, niter=niter)
  return(invisible(model))
}
model <- simulate.poisson.spike()
plot(model)
summary(model)
}
\keyword{models}
\keyword{regression}
