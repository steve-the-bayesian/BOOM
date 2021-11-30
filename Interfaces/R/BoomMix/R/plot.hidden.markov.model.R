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

plot.HiddenMarkovModel <- function(x,
                                   y = c("state.probabilities",
                                     "transition.probabilities",
                                     "initial.state.distribution",
                                     "loglikelihood",
                                     "log.likelihood",
                                     "logprior",
                                     "log.prior"),
                                   burn = 0,
                                   ...) {

  target <- try(match.arg(y), silent = TRUE)
  if (inherits(target, "try-error")) {
    ## If y fails to match one of the predefined names listed in the
    ## signature, then assume it to be the name of one of the mixture
    ## components, and we will call PlotMixtureParams for that
    ## component.

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
  } else if (target == "transition.probabilities") {
    PlotTransitionProbabilities(x, burn = burn, ...)
  } else if (target == "initial.state.distribution") {
    PlotInitialStateDistribution(x, burn = burn, ...)
  } else if (target == "state.probabilities") {
    PlotStateProbabilities(x, ...)
  } else {
    stop("plot.HiddenMarkovModel could not figure out what to plot.")
  }
  return(invisible(NULL))

}
