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
  ##     of the Dirichlet process.  Smaller values lead to more mixture
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
  ##   (TBD)
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

  class(ans) <- "DirichletProcessMvn"
  return(ans)
}

summary.DirichletProcessMvn <- function(object, burn = NULL, ...) {
  ans <- list()
  ans$cluster.size.distribution  <- sapply(ans, function(x) nrow(x$parameters$mean))
}

DpMvnClusterSizeDistribution <- function(object, burn = NULL, ...) {
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


PlotDpMvnMeans <- function(object, nclusters, ...) {
}
