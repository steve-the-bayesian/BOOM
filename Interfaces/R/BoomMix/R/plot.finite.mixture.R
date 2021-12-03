# Copyright 2021 Steven L. Scott. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

plot.FiniteMixture <- function(x,
                               y = c("state.probabilities",
                                 "mixing.weights",
                                 "loglikelihood",
                                 "log.likelihood",
                                 "logprior",
                                 "log.prior",
                                 "density"),
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
  } else if (target == "density") {
    return(PlotDensityDistribution(x, burn=burn, ...))
  } else {
    stop("PlotFiniteMixture could not figure out what to plot.")
  }
  return(invisible(NULL))
}
