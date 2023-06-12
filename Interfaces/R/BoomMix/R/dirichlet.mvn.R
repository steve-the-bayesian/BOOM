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


DirichletProcessMvn <- function(data,
                                mean.base.measure = NULL,
                                variance.base.measure = NULL,
                                concentration.parameter = 1,
                                niter,
                                ping = niter / 10,
                                seed = NULL) {
  ## Args:
  ##   data: A matrix.  Rows are observations and columns are variables.
  ##   mean.base.measure: An object of class MvnGivenSigmaMatrixPrior describing
  ##     the base measure for the mean of each mixture component.  If mu[s] and
  ##     V[s] are the mean vector and variance matrix for component s then the
  ##     model assumes mu[s] ~ N(mu0, V[s] / kappa).  This object contains mu0
  ##     and kappa.
  ##   variance.base.measure: An object of class InverseWishartPrior describing
  ##     the variance matrix V[s] for component s.  The model assumes the V[s] ~
  ##     InverseWishart(Guess, weight), parameterized so that "Guess" is the
  ##     mean of the distribution (i.e. a prior guess at the variance) and
  ##     'weight' is the degrees of freedom parameter.
  ##   concentration.parameter: A positive scalar.  The concentration parameter
  ##     of the Dirichlet process.  Larger values lead to more mixture
  ##     components.
  ##   niter:  The desired number of MCMC iterations.
  ##   ping: The frequency of status updates during the MCMC.
  ##     E.g. setting ping = 100 will print a status update every 100
  ##     MCMC iterations.
  ##   seed: An integer to use as the random seed for the underlying
  ##     C++ code.  If \code{NULL} then the seed will be set using the
  ##     clock.
  ##
  ## Returns:
  ##   The returned object is a list imbued wtih class "DirichletProcessMvn".
  ##   There are two list elements.
  ##   - log likelihood gives the log likelihood associated with each MCMC draw.
  ##       This is primarily used as a convergence diagnostic.
  ##   - parameters is a list of the drawn parameter values.  The list elements
  ##       are named according to the number of clusters that were used in each
  ##       draw.  The cluster sizes are arranged in increasing order, but need
  ##       not be congiguous.  The elements of the 'parameters' list are:
  ##     + mean - A 3-way array of draws of cluster means.  The array dimensions
  ##         are [Monte Carlo index, cluster number, data dimension].
  ##     + variance - A 4-way array of draws of cluster variances.  The array
  ##         dimensions are [Monte Carlo index, cluster number, data dimension,
  ##         data dimension].
  ##     + iteration - A vector containing the draw number of each draw in the
  ##         overall Monte Carlo sequence.  This information is needed so that
  ##         burn-in iterations can be discarded.
  data <- as.matrix(data)

  if (is.null(mean.base.measure)) {
    mean.base.measure <- MvnGivenSigmaMatrixPrior(colMeans(data), 1)
  }
  stopifnot(inherits(mean.base.measure, "MvnGivenSigmaMatrixPrior"))

  if (is.null(variance.base.measure)) {
    variance.base.measure <- InverseWishartPrior(
      var(data), ncol(data) + 1)
  }
  stopifnot(inherits(variance.base.measure, "InverseWishartPrior"))

  stopifnot(is.numeric(concentration.parameter),
    length(concentration.parameter) == 1,
    concentration.parameter > 0)

  niter <- as.integer(niter)[1]
  ping <- as.integer(round(ping))[1]
  if (!is.null(seed)) {
    seed <- as.integer(seed)[1]
  }
  ans <- .Call("boom_rinterface_fit_dirichlet_process_mvn_",
    data,
    mean.base.measure,
    variance.base.measure,
    concentration.parameter,
    niter,
    ping,
    seed,
    PACKAGE = "BoomMix")

  ans$concentration.parameter <- concentration.parameter
  ans$mean.base.measure <- mean.base.measure
  ans$variance.base.measure <- variance.base.measure

  class(ans) <- "DirichletProcessMvn"
  return(ans)
}

summary.DirichletProcessMvn <- function(object, burn = NULL, ...) {
  ## Args:
  ##   object:  A DirichletProcessMvn model object.
  ##   burn: The number of MCMC iterations to discard as burn-in.  If NULL then
  ##     SuggestBurnLogLikelihood will be called to suggest a burn-in period.
  ##
  ## A summary describing the model.
  ans <- list()
  ans$cluster.size.distribution  <- DpMvnClusterSizeDistribution(object, burn)
  return(ans)
}

DpMvnClusterSizeDistribution <- function(object, burn = NULL) {
  ## Args:
  ##   object:  A DirichletProcessMvn model object.
  ##   burn: The number of MCMC iterations to discard as burn-in.  If NULL then
  ##     SuggestBurnLogLikelihood will be called to suggest a burn-in period.
  ##
  ## Returns:
  ##   A discrete probability distribution describing the number of clusters.
  ##   The distribution may skip values with zero counts, so it may not be
  ##   contiguous.
  if (is.null(burn)) {
    burn <- SuggestBurnLogLikelihood(object$log.likelihood)
  }

  cluster.size.distribution  <- sapply(
    object$parameters, function(x) {
      indx <- x$iteration >= burn
      if (any(indx)) {
        return(nrow(x$mean[indx, , ]))
      } else {
        return(0)
      }
    })
  cluster.size.distribution <- cluster.size.distribution[
    cluster.size.distribution > 0]
  return(cluster.size.distribution / sum(cluster.size.distribution))
}

plot.DirichletProcessMvn <- function(x, y = c("means", "nclusters", "pairs",
                                              "log.likelihood", "help"), ...) {
  ## Args:
  ##   x: The DirichletProcessMvn model object to be plotted.
  ##   y: The type of plot desired.
  ##   ...: Extra arguments passed to the implementing function.
  y <- match.arg(y)
  if (y == "means") {
    PlotDpMvnMeans(x, ...)
  } else if (y == "pairs") {
    PlotDpMvnMeanPairs(x, ...)
  } else if(y == "nclusters") {
    PlotDpMvnNclusters(x, ...)
  } else if (y == "log.likelihood") {
    PlotDpMvnLoglike(x, ...)
  } else if (y == "help") {
    help("plot.DirichletProcessMvn", package = "BoomMix", help_type = "html")
  }
}

PlotDpMvnMeans <- function(model, nclusters, burn = NULL, dims = NULL, ...) {
  ## Args:
  ##   model:  The DirichletProcessMvn model object to be plotted.
  ##   nclusters:  The number of clusters of the submodel to be plotted.
  ##   burn: The number of MCMC iteration to discard as burn-in.  If NULL then
  ##     SuggestBurnLogLikelihood is called to suggest a number of burn-in
  ##     iterations.
  ##   dims: The subset of data dimensions to plot.  If NULL then everything is
  ##     plotted.
  ##   ...:  Extra arguments passed to PlotMixtureParams.
  param.list <- list()
  mu <- model$parameters[[as.character(nclusters)]]$mean
  iterations <- model$parameters[[as.character(nclusters)]]$iteration

  if (is.null(burn)) {
    burn <- SuggestBurnLogLikelihood(model$log.likelihood)
  }
  if (burn < 0) {
    burn <- 0
  }

  if (is.null(dims)) {
    ydim <- dim(mu)[3]
    dims <- 1:dim(mu)[3]
  }

  for (i in 1:nclusters) {
    mu.name <- paste0("mu.", i-1)
    param.list[[mu.name]] <- mu[iterations >= burn, i, dims]
  }
  return(PlotMixtureParams(param.list, "mu", burn = 0, ...))
}

PlotDpMvnMeanPairs <- function(model, nclusters, burn = NULL, dims = NULL,
                               gap = 0, ...) {
  ## A pairs plot showing the posterior distribution of the mean parameters.
  ##
  ## Args:
  ##   model:  The DirichletProcessMvn model object to be plotted.
  ##   nclusters:  The number of clusters of the submodel to be plotted.
  ##   burn: The number of MCMC iteration to discard as burn-in.  If NULL then
  ##     SuggestBurnLogLikelihood is called to suggest a number of burn-in
  ##     iterations.
  ##   dims: The subset of data dimensions to plot.  If NULL then everything is
  ##     plotted.
  ##   gap:  The amount of space to leave between plots, in lines of text.
  ##   ...: Extra arguments passed to 'points.'

  ## nclusters must be an integer-convertible thing of length 1.
  stopifnot(length(nclusters) == 1)
  nclusters <- as.character(as.integer(nclusters))

  if (is.null(burn)) {
    burn <- SuggestBurnLogLikelihood(model$log.likelihood)
  }
  stopifnot(is.numeric(burn), length(burn) == 1)

  if (!(nclusters %in% names(model$parameters))) {
    warning(paste0(
      "A parameter plot was requested for ", nclusters, " clusters, but the ",
      "MCMC run produced no draws with that many clusters."))
    return(NULL)
  }

  means <- model$parameters[[nclusters]]$mean
  iterations <- model$parameters[[nclusters]]$iteration
  ## means is a ndraws x nclusters x dims array.

  means <- means[iterations > burn, , ]
  if (length(means) == 0 || dim(means)[1] < 1) {
    warning(paste0("There were no iterations for ", nclusters,
      " clusters after burn-in."))
    return(NULL)
  }

  if (is.null(dims)) {
    dims <- 1:(dim(means)[3])
  }
  means <- means[, , dims]
  if (dim(means)[3] < 0) {
    stop("The requested dimensions could not be provided.")
  }

  means.dim <- dim(means)[3]
  input.par <- par()
  original.par <- par(mfrow = c(means.dim, means.dim),
    mar = rep.int(gap / 2, 4), oma = rep(5.1, 4))
  on.exit(par(original.par))

  xrange <- t(apply(means, 3, range))
  nclusters <- as.integer(nclusters)
  for (i in 1:means.dim) {
    for (j in 1:means.dim) {
      if (i == j) {
        plot(means[, 1, i],
          means[, 1, i],
          axes=FALSE,
          xlim = xrange[i, ],
          ylim = xrange[i, ],
          type = "n")
        box()
      } else {
        plot(
          means[, 1, i],
          means[, 1, j],
          axes=FALSE,
          xlim = xrange[i, ],
          ylim = xrange[j, ])
        box()
        if (nclusters > 1) {
          for (cluster_number in 2:nclusters) {
            points(
              means[, cluster_number, i],
              means[, cluster_number, j],
              pch = cluster_number,
              col=cluster_number,
              ...)
          }
        }
      }

      if (i == 1 && IsOdd(j)) {
        axis(3)
      } else if (i == means.dim && IsEven(j)) {
        axis(1)
      }

      if (j == 1 && IsOdd(i)) {
        axis(2)
      } else if (j == means.dim && IsEven(i)) {
        axis(4)
      }
    }
  }

  return(invisible(NULL))
}

PlotDpMvnNclusters <- function(object, burn = NULL, ...) {
  ## A bar graph showing the distribution of the number of clusters.
  ##
  ## Args:
  ##   object:  The DirichletProcessMvn object to plot.
  ##   burn: The number of MCMC iteration to discard as burn-in.  If NULL then
  ##     SuggestBurnLogLikelihood is called to suggest a number of burn-in
  ##     iterations.
  ##  ...: Extra arguments passed to 'barplot'.
  distribution <- DpMvnClusterSizeDistribution(object, burn=burn)
  barplot(distribution, names.arg = names(distribution), ...)
}

PlotDpMvnLoglike <- function(object, burn = 0, xlab="Iteration",
                             ylab = "Log Likelihood", ...) {
  ## Produce a time series plot of the log likelihood values achieved by the
  ## DirichletProcessMvn object.
  loglike <- object$log.likelihood
  if (is.null(burn)) {
    burn <- SuggestBurnLogLikelihood(loglike)
  }
  if (burn > 0) {
    loglike <- loglike[-(1:burn)]
  }
  plot.ts(loglike, xlab = xlab, ylab = ylab, ...)
}
