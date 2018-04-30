# Copyright 2012 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

plot.FiniteMixture <- function(x,
                               y = c("state.probabilities",
                                 "mixing.weights",
                                 "loglikelihood",
                                 "log.likelihood",
                                 "logprior",
                                 "log.prior"),
                               burn = 0,
                               ...) {
  ## S3 method for plotting the output of a finite mixture model.
  ## Args:
  ##   x:  An object inheriting from FiniteMixture.
  ##   y:  A character string indicating what to plot.  This must
  ##         expand to the name of an entry in 'x'.
  target <- try(match.arg(y), silent = TRUE)
  if (inherits(target, "try-error")) {
    PlotMixtureParams(x, y, burn = burn, ...)
    return(invisible(NULL))
  }

  if (target == "log.likelihood" || target == "loglikelihood") {
    loglike <- x$log.likelihood
    if (burn > 0) loglike <- loglike[-(1:burn)]
    plot(loglike, xlab = "MCMC Iteration",
         ylab = "log likelihood", type = "l", ...)
  } else if (target == "log.prior" || target == "logprior") {
    log.prior <- x$log.prior
    if (burn > 0) log.prior <- log.prior[-(1:burn)]
    plot(log.prior, xlab = "MCMC Iteration",
         ylab = "log prior", type = "l", ...)
  } else if (target == "mixing.weights") {
    PlotMixingWeights(x, burn = burn, ...)
  } else if (target == "state.probabilities") {
    PlotStateProbabilities(x, ...)
  } else {
    stop("PlotFiniteMixture could not figure out what to plot.")
  }
  return(invisible(NULL))
}
