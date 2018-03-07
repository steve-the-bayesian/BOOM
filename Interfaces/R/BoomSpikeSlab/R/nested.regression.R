NestedRegression <- function(response,
                             predictors,
                             group.id,
                             residual.precision.prior = NULL,
                             coefficient.prior = NULL,
                             coefficient.mean.hyperprior = NULL,
                             coefficient.variance.hyperprior = NULL,
                             suf = NULL,
                             niter,
                             ping = niter / 10,
                             sampling.method = c("ASIS", "DA"),
                             seed = NULL) {
  ## Fits a Bayesian hierarchical regression model (with a Gaussian prior) to
  ## the data provided.
  ##
  ## Args:
  ##   response:  A numeric vector.  The response variable to be modeled.
  ##   predictors: A numeric matrix of predictor variables, including an
  ##     intercept term if one is desired.  The number of rows must match
  ##     length(response).
  ##   group.id: A factor (or object that can be converted using as.factor)
  ##     naming the group to which each entry in 'response' belongs.
  ##   residual.precision.prior: An object of type SdPrior describing the prior
  ##     distribution of the residual standard deviation.
  ##   coefficient.prior: An object of class MvnPrior, or NULL.  If non-NULL
  ##     this gives the initial values of the prior distribution of the
  ##     regression coefficients in the nested regression model.  This argument
  ##     must be non-NULL if either 'coefficient.mean.hyperprior' or
  ##     'coefficient.variance.hyperprior' is NULL.
  ##   coefficient.mean.hyperprior: An object of class MvnPrior, specifying the
  ##     hyperprior distribution for the mean of 'coefficient.prior'.  This
  ##     argument can also be NULL, or FALSE.  If NULL then a default prior will
  ##     be used when learning the mean of the prior distribution.  If FALSE
  ##     then the mean of the prior distribution will not be learned; the mean
  ##     of the 'coefficient.prior' distribution will be assumed instead.
  ##   coefficient.variance.hyperprior: An object of class InverseWishartPrior,
  ##     specifying the hyperprior distribution for the variance of
  ##     'coefficient.prior'.  This argument can also be NULL, or FALSE.  If
  ##     NULL then a default prior will be used when learning the variance of
  ##     the prior distribution.  If FALSE then the variance of the prior
  ##     distribution will not be learned; the variance of the
  ##     'coefficient.prior' distribution will be assumed instead.
  ##   suf: A list, where each entry is of type RegressionSuf, giving the
  ##     sufficient statistics for each group, or NULL.  If NULL, then 'suf'
  ##     will be computed from 'response', 'predictors', and 'group.id'.  If
  ##     non-NULL then these arguments will not be accessed, in which case they
  ##     can be left unspecified.  In 'big data' problems this can be a
  ##     significant computational savings.
  ##   niter:  The desired number of MCMC iterations.
  ##   ping:  The frequency with which to print status updates.
  ##   sampling.method: Use ASIS or standard data augmentation sampling.  Both
  ##     hyperpriors must be supplied to use ASIS. If either hyperprior is set
  ##     to FALSE then the "DA" method will be used.
  ##   seed: The integer-valued seed (or NULL) to use for the C++ random number
  ##     generator.
  ##
  ## Returns:
  ##   A list containing MCMC draws from the posterior distribution of model
  ##   parameters.  Each of the following is a vector, matrix, or array, with
  ##   first index corresponding to MCMC draws, and later indices to distinct
  ##   parameters.
  ##    * coefficients: regression coefficients.
  ##    * residual.sd: the residual standard deviation from the regression
  ##        model.
  ##    * prior.mean: The posterior distribution of the coefficient means across
  ##        groups.
  ##    * prior.variance: The posterior distribution of the variance matrix
  ##        describing the distribution of regression coefficients across groups.
  if (is.null(suf)) {
    if (missing(response) || missing(predictors) || missing(group.id)) {
      stop("NestedRegression either needs a list of sufficient statistics,",
           " or a predictor matrix, response vector, and group indicators.")
    }
    suf <- .RegressionSufList(predictors, response, group.id)
  }
  stopifnot(is.list(suf))
  stopifnot(length(suf) > 0)
  stopifnot(all(sapply(suf, inherits, "RegressionSuf")))
  if (length(unique(sapply(suf, function(x) ncol(x$xtx)))) != 1) {
    stop("All RegressionSuf objects must have the same dimensions.")
  }
  xdim <- ncol(suf[[1]]$xtx)
  sampling.method <- match.arg(sampling.method)
  ##----------------------------------------------------------------------
  ## Check that the priors are from the right families.
  if (is.null(residual.precision.prior)) {
    residual.precision.prior <- .DefaultNestedRegressionResidualSdPrior(suf)
  }
  stopifnot(inherits(residual.precision.prior, "SdPrior"))

  ##----------------------------------------------------------------------
  if (is.logical(coefficient.mean.hyperprior) &&
      coefficient.mean.hyperprior == FALSE) {
    sampling.method <- "DA"
    coefficient.mean.hyperprior <- NULL
  } else {
    if (is.null(coefficient.mean.hyperprior)) {
      coefficient.mean.hyperprior <- .DefaultNestedRegressionMeanHyperprior(suf)
    }
    stopifnot(inherits(coefficient.mean.hyperprior, "MvnPrior"))
    stopifnot(length(coefficient.mean.hyperprior$mean) == xdim)
  }

  ##----------------------------------------------------------------------
  if (is.logical(coefficient.variance.hyperprior) &&
                 coefficient.variance.hyperprior == FALSE) {
    coefficient.variance.hyperprior <- NULL
    sampling.method <- "DA"
  } else if (is.null(coefficient.variance.hyperprior)) {
    coefficient.variance.hyperprior <-
      .DefaultNestedRegressionVarianceHyperprior(suf)
    stopifnot(inherits(coefficient.variance.hyperprior, "InverseWishartPrior"),
              ncol(coefficient.variance.hyperprior$variance.guess) == xdim)
  }
  ##--------------------------------------------------------------------------
  ## If either hyperprior is non-NULL then coefficient.prior must be supplied.
  ## Otherwise, create a default value.
  if (!is.null(coefficient.mean.hyperprior) &&
      !is.null(coefficient.variance.hyperprior)) {
    if (is.null(coefficient.prior)) {
      coefficient.prior <- MvnPrior(rep(0, xdim), diag(rep(1, xdim)))
    }
  }
  stopifnot(inherits(coefficient.prior, "MvnPrior"))
  ##----------------------------------------------------------------------
  ## Check remaining arguments.
  stopifnot(is.numeric(niter),
            length(niter) == 1,
            niter > 0)
  stopifnot(is.numeric(ping),
            length(ping) == 1)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }
  ##----------------------------------------------------------------------
  ## Have C++ do the heavy lifting.
  ans <- .Call("boom_nested_regression_wrapper",
               suf,
               coefficient.prior,
               coefficient.mean.hyperprior,
               coefficient.variance.hyperprior,
               residual.precision.prior,
               as.integer(niter),
               as.integer(ping),
               sampling.method,
               seed)
  ans$priors <- list(
      coefficient.prior = coefficient.prior,
      coefficient.mean.hyperprior = coefficient.mean.hyperprior,
      coefficient.variance.hyperprior = coefficient.variance.hyperprior,
      residual.precision.prior = residual.precision.prior)

  ##----------------------------------------------------------------------
  ## Slap on a class, and return the answer.
  class(ans) <- "NestedRegression"
  return(ans)
}
###======================================================================
.RegressionSufList <- function(predictors, response, group.id) {
  ## Args:
  ##   predictors:  A matrix of predictor variables
  ##   response: A numeric response variable with length matching
  ##     nrow(predictors).
  ##   group.id:  A factor with length matching 'response'.
  ##
  ## Returns:
  ##   A list of RegressionSuf objects, one for each unique value in group.id.
  ##   Each list element contains the sufficient statistics for a regression
  ##   model for the subset of data corresponding to that value of group.id.
  stopifnot(is.numeric(response))
  stopifnot(is.matrix(predictors),
            nrow(predictors) == length(response))
  group.id <- as.factor(group.id)
  stopifnot(length(group.id) == length(response))

  MakeRegSuf <- function(data) {
    return(RegressionSuf(X = as.matrix(data[,-1]),
                         y = as.numeric(data[,1])))
  }
  return(by(as.data.frame(cbind(response, predictors)),
            group.id,
            MakeRegSuf))
}
###======================================================================
.CollapseRegressionSuf <- function(reg.suf.list) {
  ## Args:
  ##   reg.suf.list: A list of objects of class RegressionSuf.  Elements must be
  ##     of the same dimension (i.e. all regressions must have the same number
  ##     of predictors).
  ## Returns:
  ##   A single RegressionSuf object formed by accumulating the sufficient
  ##   statistics from each list element.
  stopifnot(is.list(reg.suf.list),
            length(reg.suf.list) > 0,
            all(sapply(reg.suf.list, inherits, "RegressionSuf")))
  if (length(reg.suf.list) == 1){
    return(reg.suf.list[[1]])
  }
  xtx <- reg.suf.list[[1]]$xtx
  xty <- reg.suf.list[[1]]$xty
  yty <- reg.suf.list[[1]]$yty
  n <- reg.suf.list[[1]]$n
  xsum <- reg.suf.list[[1]]$xbar * n
  for (i in 2:length(reg.suf.list)) {
    xtx <- xtx + reg.suf.list[[i]]$xtx
    xty <- xty + reg.suf.list[[i]]$xty
    yty <- yty + reg.suf.list[[i]]$yty
    n <- n + reg.suf.list[[i]]$n
    xsum <- xsum + reg.suf.list[[i]]$xbar * reg.suf.list[[i]]$n
  }
  xbar <- xsum / n
  return(RegressionSuf(xtx = xtx, xty = xty, yty = yty, n = n, xbar = xbar))
}
###======================================================================
.ResidualVariance <- function(suf) {
  ## Args:
  ##   suf:  Sufficient statistics for a regression problem.
  ## Returns:
  ##   The maximum likelihood (biased, but consistent) estimate of the residual
  ##   variance in the regression problem.
  stopifnot(inherits(suf, "RegressionSuf"))
  sse <- as.numeric(suf$yty - t(suf$xty) %*% solve(suf$xtx, suf$xty))
  df.model <- ncol(suf$xtx)
  return(sse / (suf$n - df.model))
}
###======================================================================
.DefaultNestedRegressionMeanHyperprior <- function(suf) {
  ## Args:
  ##   suf: A list of RegressionSuf sufficient statistics.
  ## Returns:
  ##   An object of class MvnPrior that can serve as the default prior for the
  ##   prior mean parameters in a NestedRegression model.
  suf <- .CollapseRegressionSuf(suf)
  ## If xtx is full rank then center the prior on the OLS estimate beta-hat.
  beta.hat <- tryCatch(solve(suf$xtx, suf$xty))
  if (is.numeric(beta.hat)) {
    ## If the OLS estimate is computable, use it to center and scale the prior
    ## mean for the prior mean parameters.
    return(MvnPrior(beta.hat,  .ResidualVariance(suf) * solve(suf$xtx / suf$n)))
  } else {
    ## If the OLS estimate is not computable (because the xtx matrix is not full
    ## rank) then center the prior mean at zero with what is hopefully a big
    ## variance.
    xdim <- length(suf$xty)
    zero <- rep(0, xdim);
    V <- diag(rep(1000), xdim)
    return(MvnPrior(zero, V))
  }
}
###======================================================================
.DefaultNestedRegressionVarianceHyperprior <- function(suf) {
  ## Args:
  ##   suf: A list of RegressionSuf sufficient statistics.
  ## Returns:
  ##   An object of class InverseWishartPrior that can serve as the default
  ##   prior for the prior variance parameters in a NestedRegression model.
  number.of.groups <- length(suf)
  suf <- .CollapseRegressionSuf(suf)

  ## Shrink the variance towards a small but realistic number.  The variance of
  ## the grand mean is sigsq / X'X.  It is also roughly Var(beta_g) /
  ## number.of.groups, so let's say the prior variance of beta_g is
  ## number.of.groups * sigsq / X'X.
  variance.guess <- .ResidualVariance(suf) * number.of.groups * solve(suf$xtx)
  variance.guess.weight <- ncol(variance.guess) + 1
  return(InverseWishartPrior(variance.guess, variance.guess.weight))
}

.DefaultNestedRegressionResidualSdPrior <- function(suf) {
  ## Args:
  ##   suf: A list of RegressionSuf sufficient statistics.
  ## Returns:
  ##   An object of class SdPrior that can serve as the default prior for the
  ##   residual standard deviation in a NestedRegression model.
  ## Details:
  ##   Shrinks (with one degree of freedom) the residual variance towards the
  ##   residual variance from an OLS model based on pooled data.
  suf <- .CollapseRegressionSuf(suf)
  variance.guess <- .ResidualVariance(suf)
  return(SdPrior(sqrt(variance.guess), 1))
}
