CheckFunction <- function(u, quantile) {
  stopifnot(is.numeric(u))
  stopifnot(is.numeric(quantile),
            length(quantile) == 1,
            quantile > 0,
            quantile < 1)
  negative <- u < 0
  ans <- quantile * u
  ans[negative] <- -(1 - quantile) * u[negative]
  return(ans)
}

qreg.spike <- function(
    formula,
    quantile,
    niter,
    ping = niter / 10,
    nthreads = 0,
    data,
    subset,
    prior = NULL,
    na.action = options("na.action"),
    contrasts = NULL,
    drop.unused.levels = TRUE,
    initial.value = NULL,
    seed = NULL,
    ...) {
  ## Uses Bayesian MCMC to fit a quantile regression model with a
  ## spike-and-slab prior.
  ##
  ## Args:
  ##   formula: Model formula, as would be passed to 'glm', specifying
  ##     the maximal model (i.e. the model with all predictors
  ##     included).
  ##   quantile: The quantile of the response variable to be targeted
  ##     by the regression.  A scalar between 0 and 1.
  ##   niter:  desired number of MCMC iterations
  ##   ping: If positive, then print a status update every 'ping' MCMC
  ##     iterations.
  ##   nthreads:  The number of threads to use when imputing latent data.
  ##   data:  optional data.frame containing the data described in 'formula'
  ##   subset: an optional vector specifying a subset of observations
  ##     to be used in the fitting process.
  ##   prior: An optional list such as that returned from
  ##     SpikeSlabPrior.  If missing, SpikeSlabPrior
  ##     will be called with the remaining arguments.
  ##   na.action: A function which indicates what should happen when
  ##     the data contain ‘NA’s.  The default is set by the
  ##     ‘na.action’ setting of ‘options’, and is ‘na.fail’ if that is
  ##     unset.  The ‘factory-fresh’ default is ‘na.omit’.  Another
  ##     possible value is ‘NULL’, no action.  Value ‘na.exclude’ can
  ##     be useful.
  ##   contrasts: An optional list. See the ‘contrasts.arg’ of
  ##     ‘model.matrix.default’.  An optional list.
  ##   drop.unused.levels: should factor levels that are unobserved be
  ##     dropped from the model?
  ##   initial.value: Initial value of quantile regression
  ##     coefficients for the MCMC algorithm.  Can be given as a
  ##     numeric vector, a 'qreg.spike' object, or a 'glm' object.
  ##     If a 'qreg.spike' object is used for initialization, it is
  ##     assumed to be a previous MCMC run to which 'niter' futher
  ##     iterations should be added.  If a 'glm' object is supplied,
  ##     its coefficients will be used as the initial values in the
  ##     MCMC simulation.
  ##   seed: Seed to use for the C++ random number generator.  NULL or
  ##     an int.  If NULL, then the seed will be taken from the global
  ##     .Random.seed object.
  ##   ... : parameters to be passed to SpikeSlabPrior
  ##
  ## Returns:
  ##   An object of class 'qreg.spike', which is a list containing the
  ##   following values:
  ##   beta: A 'niter' by 'ncol(X)' matrix of regression coefficients
  ##     many of which may be zero.  Each row corresponds to an MCMC
  ##     iteration.
  ##   prior:  The prior that was used to fit the model.
  ##  In addition, the returned object contains sufficient details for
  ##  the call to model.matrix in the predict.qreg.spike method.
  quantile.arg <- quantile
  stopifnot(is.numeric(quantile.arg),
            length(quantile.arg) == 1,
            quantile.arg > 0,
            quantile.arg < 1)

  has.data <- !missing(data)
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "subset", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- drop.unused.levels
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  response <- model.response(mf, "numeric")

  predictor.matrix <- model.matrix(mt, mf, contrasts)
  if (is.null(prior)) {
    prior <- SpikeSlabPrior(predictor.matrix, response, ...)
  }

  if (!is.null(initial.value)) {
    if (inherits(initial.value, "qreg.spike")) {
      stopifnot(colnames(initial.value$beta) == colnames(predictor.matrix))
      beta0 <- as.numeric(tail(initial.value$beta, 1))
    } else if (inherits(initial.value, "glm")) {
      stopifnot(colnames(initial.value$beta) == colnames(predictor.matrix))
      beta0 <- coef(initial.value)
    } else if (is.numeric(initial.value)) {
      stopifnot(length(initial.value) == ncol(predictor.matrix))
      beta0 <- initial.value
    } else {
      stop("initial.value must be a 'qreg.spike' object, a 'glm' object,",
           "or a numeric vector")
    }
  } else {
    ## No initial value was supplied.  The initial condition is set so
    ## that all slopes are zero.  The intercept is set to the targeted
    ## quantile of the response variable.
    beta0 <- rep(0, ncol(predictor.matrix))
    beta0[1] <- quantile(response, quantile, na.rm = TRUE)
  }

  if (!is.null(seed)) {
    seed <- as.integer(seed)
    stopifnot(length(seed) == 1)
  }

  ans <- .Call("analysis_common_r_quantile_regression_spike_slab",
               predictor.matrix,
               response,
               quantile.arg,
               prior,
               as.integer(niter),
               as.integer(ping),
               as.integer(nthreads),
               beta0,
               seed)

  variable.names <- colnames(predictor.matrix)
  if (!is.null(variable.names)) {
    colnames(ans$beta) <- variable.names
  }
  ans$prior <- prior

  ## The stuff below will be needed by predict.qreg.spike.
  ans$contrasts <- attr(predictor.matrix, "contrasts")
  ans$xlevels <- .getXlevels(mt, mf)
  ans$call <- cl
  ans$terms <- mt

  if (!is.null(initial.value) && inherits(initial.value, "qreg.spike")) {
    ans$beta <- rbind(initial.value$beta, ans$beta)
  }

  ## (niter x p) %*% (p x n)
  prediction.distribution <- ans$beta %*% t(predictor.matrix)
  residual.distribution <-
    matrix(rep(response, each = nrow(ans$beta)),
           nrow = nrow(ans$beta)) - prediction.distribution

  ans$log.likelihood <-
    -.5 * rowSums(CheckFunction(residual.distribution, quantile))

  if (has.data) {
    ## Note, if a data.frame was passed as an argument to this function
    ## then saving the data frame will be cheaper than saving the
    ## model.frame.
    ans$training.data <- data
  } else {
    ## If the model was called with a formula referring to objects in
    ## another environment, then saving the model frame will capture
    ## these variables so they can be used to recreate the design
    ## matrix.
    ans$training.data <- mf
  }

  ## Make the answer a class, so that the right methods will be used.
  class(ans) <- c("qreg.spike", "glm.spike", "lm.spike")
  return(ans)
}

predict.qreg.spike <- function(object, newdata, burn = 0,
                               na.action = na.pass, ...) {
  ## Prediction method for qreg.spike.
  ## Args:
  ##   object: object of class "qreg.spike" returned from the
  ##     qreg.spike function
  ##   newdata: A data frame including variables with the same names
  ##     as the data frame used to fit 'object'.
  ##   burn: The number of MCMC iterations in 'object' that should be
  ##     discarded.  If burn < 0 then all iterations are kept.
  ##   ...: unused, but present for compatibility with generic predict().
  ## Returns:
  ##   A matrix of predictions, with each row corresponding to a row
  ##   in newdata, and each column to an MCMC iteration.
  predictors <- GetPredictorMatrix(object, newdata, na.action = na.action, ...)
  beta <- object$beta
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  return(predictors %*% t(beta))
}

plot.qreg.spike <- function(
    x,
    y = c("inclusion", "coefficients", "scaled.coefficients", "size", "help"),
    burn = SuggestBurnLogLikelihood(x$log.likelihood),
    ...) {
  y <- match.arg(y)
  if (y == "inclusion") {
    return(PlotMarginalInclusionProbabilities(x$beta, burn = burn, ...))
  } else if (y == "coefficients") {
    return(PlotLmSpikeCoefficients(x$beta, burn = burn, ...))
  } else if (y == "scaled.coefficients") {
    scale.factors <- apply(model.matrix(x), 2, sd)
    if (abs(scale.factors[1]) < 1e-8 && names(scale.factors)[1] == "(Intercept)") {
      scale.factors[1] <- 1.0
    }
    return(PlotLmSpikeCoefficients(x$beta, burn = burn,
                                   scale.factors = scale.factors, ...))
  } else if (y == "size") {
    return(PlotModelSize(x$beta, ...))
  } else if (y == "help") {
    help("plot.qreg.spike", package = "BoomSpikeSlab", help_type = "html")
  }
}
