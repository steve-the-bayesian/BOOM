% Copyright 2016 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{probit.spike}
\Rdversion{1.1}
\alias{probit.spike}
\title{
  Spike and slab probit regression
}
\description{
  MCMC algorithm for logistic regression models with a 'spike-and-slab'
  prior that places some amount of posterior probability at zero for a
  subset of the regression coefficients.
}
\usage{
probit.spike(formula,
             niter,
             data,
             subset,
             prior = NULL,
             na.action = options("na.action"),
             contrasts = NULL,
             drop.unused.levels = TRUE,
             initial.value = NULL,
             ping = niter / 10,
             clt.threshold = 5,
             proposal.df = 3,
             sampler.weights = c(.5, .5),
             seed = NULL,
             ...)
}
\arguments{

  \item{formula}{Formula for the maximal model (with all variables
    included).  This is parsed the same way as a call to
    \code{\link{glm}}, but no \code{family} argument is needed.  Like
    \code{\link{glm}}, a two-column input format (success-count,
    failure-count) can be used for the reponse.  Otherwise, the response
    variable can be a logical or numeric vector.  If a single-column
    response is numeric, then a positive value indicates a "success".  }

  \item{niter}{ The number of MCMC iterations to run.  Be sure to
    include enough so you can throw away a burn-in set.  }

  \item{data}{ An optional data frame, list or environment (or object
    coercible by 'as.data.frame' to a data frame) containing the
    variables in the model.  If not found in 'data', the variables are
    taken from 'environment(formula)', typically the environment from
    which probit.spike' is called.  }

  \item{subset}{ An optional vector specifying a subset of observations
    to be used in the fitting process.  }

  \item{prior}{An object inheriting from \code{\link{LogitPrior}} and
    \code{\link{SpikeSlabPriorBase}}.  If \code{prior} is supplied it
    will be used.  Otherwise a prior distribution will constructed by
    calling \code{\link{LogitZellnerPrior}} with the remaining
    arguments. Despite the name, LogitPrior objects are appropriate for
    Probit models.}

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
    coefficients will be used), or a \code{\link{probit.spike}} object.
    If a \code{\link{probit.spike}} object is supplied, it is assumed to
    be from a previous MCMC run for which \code{niter} additional draws
    are desired.  If a \code{\link{glm}} object is supplied then its
    coefficients will be used as the initial values for the simulation.
    }

  \item{ping}{If positive, then print a status update to the console
    every \code{ping} MCMC iterations.}

  \item{clt.threshold}{When the model is presented with binomial data
    (i.e. when the response is a two-column matrix) the data
    augmentation algorithm can be made more efficient by updating a
    single, asymptotically normal scalar quantity for each unique value
    of the predictors.  The asymptotic result will be used whenever the
    number of successes or failures exceeds \code{clt.threshold}.
    }

  \item{proposal.df}{The degrees of freedom parameter to use in
    Metropolis-Hastings proposals.  See details.}

  \item{sampler.weights}{A two-element vector giving the probabilities
    of drawing from the two base sampling algorithm.  The first element
    refers to the spike and slab algorithm.  The second refers to the
    tailored independence Metropolis sampler.  TIM is usually faster
    mixing, but cannot change model dimension.}

  \item{seed}{Seed to use for the C++ random number generator.  It
    should be \code{NULL} or an int.  If \code{NULL} the seed value will
    be taken from the global \code{\link{.Random.seed}} object.}

  \item{\dots}{
    Extra arguments to be passed to \code{\link{LogitZellnerPrior}}.
  }

}
\value{

  Returns an object of class \code{probit.spike}, which inherits from
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

  Model parameters are updated using a composite of two
  Metropolis-Hastings updates.  A data augmentation algorithm (Albert
  and Chib 1993) updates the entire parameter vector at once, but can
  mix slowly.

  The second algorithm is an independence Metropolis sampler centered on
  the posterior mode with variance determined by posterior information
  matrix (Fisher information plus prior information).  If
  \code{proposal.df > 0} then the tails of the proposal are inflated so
  that a multivariate T proposal is used instead.

  At each iteration, one of the three algorithms is chosen at random.
  The auxiliary mixture sampler is the only one that can change the
  dimension of the coefficient vector.  The MH algorithm only updates
  the coefficients that are currently nonzero.  }

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}},
  \code{\link{plot.probit.spike}},
  \code{\link{PlotProbitSpikeFitSummary}}
  \code{\link{PlotProbitSpikeResiduals}}
  \code{\link{summary.logit.spike}},
  \code{\link{predict.logit.spike}}.
}

\examples{
if (requireNamespace("MASS")) {
  data(Pima.tr, package = "MASS")
  data(Pima.te, package = "MASS")
  pima <- rbind(Pima.tr, Pima.te)
  model <- probit.spike(type == "Yes" ~ ., data = pima, niter = 500)
  plot(model)
  plot(model, "fit")
  plot(model, "residuals")
  plot(model, "size")
  summary(model)
}
}
\keyword{models}
\keyword{regression}
