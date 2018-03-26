ShrinkageRegression <- function(response,
                                predictors,
                                coefficient.groups,
                                residual.precision.prior = NULL,
                                suf = NULL,
                                niter,
                                ping = niter / 10,
                                seed = NULL) {
  ## Fit a Bayesian regression model with a shrinkage prior on the coefficient.
  ## The model is
  ##
  ##                    y[i] ~ N(x[i,] %*% beta, sigma^2)
  ##             1 / sigma^2 ~ Gamma(df/2, ss/2)
  ##          group(beta, 1) ~ N(b1, v1)
  ##          group(beta, 2) ~ N(b2, v2)
  ##          ....
  ##
  ##  In this notation, group(beta, k) ~ N(bk, vk) indicates that the subset of
  ##  coefficients in group k are a priori independent draws from the specified
  ##  normal distribution. In addition, each subset-level prior may include a
  ##  hyperprior, in which case the subset-level prior parameters will be
  ##  updated as part of the MCMC.  The hyperprior has the form of independent
  ##  priors on the mean and precision parameters.
  ##
  ##                      bi ~ N(prior.mean, prior.variance)
  ##                1.0 / vi ~ Chisq(df, guess.at.sd)
  ##
  ## Args:
  ##   response:  The numeric vector of responses.
  ##   predictors:  The matrix of predictors, including an intercept term, if
  ##     desired.
  ##   coefficient.groups: A list of objects of type CoefficientGroup, defining
  ##     the pattern in which the coefficients should be shrunk together.  Each
  ##     coefficient must belong to exactly one CoefficientGroup.
  ##   residual.precision.prior: An object of type SdPrior describing the prior
  ##     distribution of the residual standard deviation.
  ##   suf: An object of class RegressionSuf containing the sufficient
  ##     statistics for the regression model.  If this is NULL then it will be
  ##     computed from 'response' and 'predictors'.  If it is supplied then
  ##     'response' and 'predictors' are not used and can be left missing.
  ##   niter:  The desired number of MCMC iterations.
  ##   ping:  The frequency with which to print status updates.
  ##   seed: The integer-valued seed (or NULL) to use for the C++ random number
  ##     generator.
  ##
  ## Returns:
  ##   A list containing MCMC draws from the posterior distribution of model
  ##   parameters.  Each of the following is a matrix, with rows corresponding
  ##   to MCMC draws, and columns to distinct parameters.
  ##    * coefficients: regression coefficients.
  ##    * residual.sd: the residual standard deviation from the regression
  ##        model.
  ##    * group.means: The posterior distribution of the mean of each
  ##        coefficient group.  If no mean hyperprior was assigned to a
  ##        particular group, then the value here will be a constant (the values
  ##        supplied by the 'prior' argument to 'CoefficientGroup' for that
  ##        group).
  ##    * group.sds: The posterior distribution of the standard deviation of
  ##        each coefficient group.  If no sd.hyperprior was assigned to a
  ##        particular group, then the value here will be a constant (the values
  ##        supplied by the 'prior' argument to 'CoefficientGroup' for that
  ##        group).

  if (is.null(suf)) {
    ## No need to touch response or predictors if sufficient statistics are
    ## supplied directly.
    stopifnot(is.numeric(response))
    stopifnot(is.matrix(predictors),
              nrow(predictors) == length(response))
    stopifnot(is.list(coefficient.groups),
              all(sapply(coefficient.groups, inherits, "CoefficientGroup")))
    suf <- RegressionSuf(X = predictors, y = response)
  }
  stopifnot(inherits(suf, "RegressionSuf"))
  stopifnot(is.list(coefficient.groups))
  all.indices <- sort(unlist(lapply(coefficient.groups, function(x) x$indices)))
  xdim <- ncol(suf$xtx)
  if (any(all.indices != 1:xdim)) {
    if (any(all.indices > xdim)) {
      stop("One or more indices were larger than the available ",
           "number of predictors.")
    } else if (any(all.indices <= 0)) {
      stop("All indices must be 1 or larger.")
    } else if (all.indices != unique(all.indices)) {
      stop("Each index can only appear in one group.")
    } else {
      omitted <- !(all.indices %in% 1:xdim)
      omitted.indices <- all.indices[omitted]
      if (length(omitted.indices) > 10) {
        msg <- paste("There were ", length(omitted.indices),
                     " indices omitted from index groups.")
        stop(msg)
      } else if (length(omitted.indices > 1)) {
        msg <- paste("The following indices were omitted from a ",
                     "coefficient groups: \n")
        msg <- paste(msg, paste(omitted.indices, collapse = " "), "\n")
        stop(msg)
      } else {
        msg <- paste("Index ", omitted.indices,
                     " was not present in any coefficient groups.\n")
        stop(msg)
      }
    }
  }  ## done checking index groups

  if (is.null(residual.precision.prior)) {
    residual.precision.prior <- SdPrior(1, 1)
  }
  stopifnot(inherits(residual.precision.prior, "SdPrior"))

  stopifnot(is.numeric(niter),
            length(niter) == 1,
            niter > 0)
  stopifnot(is.numeric(ping),
            length(ping) == 1)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  ans <- .Call("boom_shrinkage_regression_wrapper",
               suf,
               coefficient.groups,
               residual.precision.prior,
               as.integer(niter),
               as.integer(ping),
               seed)

  class(ans) <- "ShrinkageRegression"
  return(ans)
}

CoefficientGroup <- function(indices,
                             mean.hyperprior = NULL,
                             sd.hyperprior = NULL,
                             prior = NULL) {
  ## Args:
  ##   indices: A vector of integers giving the positions of the regression
  ##     coefficients that should be viewed as exchangeable.
  ##   mean.hyperprior: A NormalPrior object describing the hyperprior
  ##     distribution for the average coefficient.
  ##   sd.hyperprior: An SdPrior object describing the hyperprior distribution
  ##     for the standard deviation of the coefficients.
  ##   prior: An object of type NormalPrior giving the initial value of the
  ##     distribution describing the collection of coefficients in this group.
  ##     If either hyperprior is NULL then the corresponding prior parameter
  ##     will not be updated.  If both hyperpriors are non-NULL then this
  ##     parameter can be left unspecified.
  ##
  ## Returns:
  ##   An object (list) containing the arguments, with values checked for
  ##   legality, and with names as expected by the underlying C++ code.
  ##
  ## Details:
  ##   The model for the coefficients in this group is that they are independent
  ##   draws from N(b0, sigma^2 / kappa), where sigma^2 is the residual variance
  ##   from the regression model.  The hyperprior distribution for this model is
  ##   b0 ~ mean.hyperprior and kappa ~ shrinkage.hyperprior, independently.
  stopifnot(is.numeric(indices),
            length(unique(indices)) == length(indices),
            all(indices >= 1))
  if (!is.null(mean.hyperprior)) {
    stopifnot(inherits(mean.hyperprior, "NormalPrior"))
  }
  if (!is.null(sd.hyperprior)) {
    stopifnot(inherits(sd.hyperprior, "SdPrior"))
  }
  if (is.null(prior) && (is.null(mean.hyperprior) || is.null(sd.hyperprior))) {
    stop("If either hyperprior is NULL, then an initial prior distribution ",
         "must be supplied.")
  }
  if (!is.null(prior)) {
    stopifnot(inherits(prior, "NormalPrior"))
  }
  ans <- list(indices = as.integer(indices),
              mean.hyperprior = mean.hyperprior,
              sd.hyperprior = sd.hyperprior,
              prior = prior)
  class(ans) <- "CoefficientGroup"
  return(ans)
}
