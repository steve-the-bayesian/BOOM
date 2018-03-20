% Copyright 2010 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{logit.spike}
\Rdversion{1.1}
\alias{logit.spike}
\title{
  Spike and slab logistic regression
}
\description{
  MCMC algorithm for logistic regression models with a 'spike-and-slab'
  prior that places some amount of posterior probability at zero for a
  subset of the regression coefficients.
}
\usage{
logit.spike(formula,
            niter,
            data,
            subset,
            prior = NULL,
            na.action = options("na.action"),
            contrasts = NULL,
            drop.unused.levels = TRUE,
            initial.value = NULL,
            ping = niter / 10,
            nthreads = 0,
            clt.threshold = 2,
            mh.chunk.size = 10,
            proposal.df = 3,
            sampler.weights = c("DA" = .333, "RWM" = .333, "TIM" = .333),
            seed = NULL,
            ...)
}
\arguments{

  \item{formula}{ formula for the maximal model (with all variables
    included), this is parsed the same way as a call to
    \code{\link{glm}}, but no \code{family} argument is needed.  Like
    \code{\link{glm}}, a two-column input format (success-count,
    failure-count).  Otherwise, the response variable can be a logical
    or numeric vector.  If numeric, then values >0 indicate a
    "success".  }

  \item{niter}{ The number of MCMC iterations to run.  Be sure to
    include enough so you can throw away a burn-in set.  }

  \item{data}{ An optional data frame, list or environment (or object
    coercible by 'as.data.frame' to a data frame) containing the
    variables in the model.  If not found in 'data', the variables are
    taken from 'environment(formula)', typically the environment from
    which logit.spike' is called.  }

  \item{subset}{ An optional vector specifying a subset of observations
    to be used in the fitting process.  }

  \item{prior}{A n object inheriting from \code{\link{LogitPrior}} and
    \code{\link{SpikeSlabPriorBase}}.  If \code{prior} is supplied it
    will be used.  Otherwise a prior distribution will constructed by
    calling \code{\link{LogitZellnerPrior}} with the remaining
    arguments. }

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
    coefficients will be used), or a \code{\link{logit.spike}} object.
    If a \code{\link{logit.spike}} object is supplied, it is assumed to
    be from a previous MCMC run for which \code{niter} additional draws
    are desired.  If a \code{\link{glm}} object is supplied then its
    coefficients will be used as the initial values for the simulation.
    }

  \item{ping}{If positive, then print a status update to the console
    every \code{ping} MCMC iterations.}

  \item{nthreads}{The number of CPU-threads to use for data
    augmentation.  There is some small overhead to stopping and starting
    threads.  For small data sets, thread overhead will make it faster
    to run single threaded.  For larger data sets multi-threading can
    speed things up substantially.  This is all machine dependent, so
    please experiment.}

  \item{clt.threshold}{When the model is presented with binomial data
    (i.e. when the response is a two-column matrix) the data
    augmentation algorithm can be made more efficient by updating a
    single, asymptotically normal scalar quantity for each unique value
    of the predictors.  The asymptotic result will be used whenever the
    number of successes or failures exceeds \code{clt.threshold}.
    }

  \item{mh.chunk.size}{The maximum number of coefficients to draw in a
    single "chunk" of a Metropolis-Hastings update.  See details. }

  \item{proposal.df}{The degrees of freedom parameter to use in
    Metropolis-Hastings proposals.  See details.}

  \item{sampler.weights}{The proportion of MCMC iterations spent in each
    of the three algorithms described in the Details section. This must
    be a vector of length 3, with names "DA", "RWM" and "TIM",
    containing non-negative elements that sum to (within numerical
    error .999 or 1.001 are okay).}

  \item{seed}{Seed to use for the C++ random number generator.  It
    should be \code{NULL} or an int.  If \code{NULL} the seed value will
    be taken from the global \code{\link{.Random.seed}} object.}

  \item{\dots}{
    Extra arguments to be passed to \code{\link{SpikeSlabPrior}}.
  }

}
\value{

  Returns an object of class \code{logit.spike}, which inherits from
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

  Model parameters are updated using a composite of three
  Metropolis-Hastings updates.  An auxiliary mixture sampling algorithm
  (Tuchler 2008) updates the entire parameter vector at once, but can
  mix slowly.

  The second algorithm is a random walk Metropolis update based on a
  multivariate T proposal with \code{proposal.df} degrees of freedom.
  If \code{proposal.df} is nonpositive then a Gaussian proposal is used.
  The variance of the proposal distribution is based on the Fisher
  information matrix evaluated at the current draw of the coefficients.

  The third algorithm is an independence Metropolis sampler centered on
  the posterior mode with variance determined by posterior information
  matrix (Fisher information plus prior information).  If
  \code{proposal.df > 0} then the tails of the proposal are inflated so
  that a multivariate T proposal is used instead.

  For either of the two MH updates, at most \code{mh.chunk.size}
  coefficients will be updated at a time.  At each iteration, one of the
  three algorithms is chosen at random.  The auxiliary mixture sampler
  is the only one that can change the dimension of the coefficient
  vector.  The MH algorithms only update the coefficients that are
  currently nonzero.
}


\references{
  Tuchler (2008), "Bayesian Variable Selection for Logistic Models Using
  Auxiliary Mixture Sampling", \emph{Journal of Computational and
  Graphical Statistics}, \bold{17} 76 -- 94.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}},
  \code{\link{plot.logit.spike}},
  \code{\link{PlotLogitSpikeFitSummary}}
  \code{\link{PlotLogitSpikeResiduals}}
  \code{\link{summary.logit.spike}},
  \code{\link{predict.logit.spike}}.
}

\examples{
if (requireNamespace("MASS")) {
  data(Pima.tr, package = "MASS")
  data(Pima.te, package = "MASS")
  pima <- rbind(Pima.tr, Pima.te)
  model <- logit.spike(type == "Yes" ~ ., data = pima, niter = 500)
  plot(model)
  plot(model, "fit")
  plot(model, "residuals")
  plot(model, "size")
  summary(model)
}
}
\keyword{models}
\keyword{regression}
