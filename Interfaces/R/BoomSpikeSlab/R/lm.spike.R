lm.spike <- function(formula,
                     niter,
                     data,
                     subset,
                     prior = NULL,
                     error.distribution = c("gaussian", "student"),
                     contrasts = NULL,
                     drop.unused.levels = TRUE,
                     bma.method = c("SSVS", "ODA"),
                     oda.options = list(
                         fallback.probability = 0.0,
                         eigenvalue.fudge.factor = 0.01),
                     ping = niter / 10,
                     seed = NULL,
                     ...) {
  ## Uses Bayesian MCMC to fit a linear regression model with a
  ## spike-and-slab prior.
  ##
  ## Args:
  ##   formula: Model formula, as would be passed to 'lm', specifying
  ##     the maximal model (i.e. the model with all predictors
  ##     included).
  ##   niter:  Desired number of MCMC iterations
  ##   data:  Optional data.frame containing the data described in 'formula'
  ##   subset: An optional vector specifying a subset of observations
  ##     to be used in the fitting process.
  ##   prior: An object of class SpikeSlabPrior or
  ##     IndependentSpikeSlabPrior.  If missing, a default prior will
  ##     be generated using the arguments passed as ... .  If
  ##     'bma.method' is SSVS then either a SpikeSlabPrior or
  ##     IndependentSpikeSlabPrior can be used.  (A SpikeSlabPrior will
  ##     be used as the default).  If 'bma.method' is ODA then an
  ##     IndependentSpikeSlabPrior is required.
  ##   error.distribution: Specify either Gaussian or Student T
  ##     errors.  If the error distribution is student then the prior
  ##     must be a StudentSpikeSlabPrior.
  ##   contrasts: An optional list. See the 'contrasts.arg' of
  ##     ‘model.matrix.default’.
  ##   drop.unused.levels: logical.  Should unobserved factor levels
  ##     be dropped from the model?
  ##   bma.method: The method to use for Bayesian model averaging.
  ##     "SSVS" is stochastic search variable selection, which is the
  ##     classic approach from George and McCulloch (1997).  "ODA" is
  ##     orthoganal data augmentation, from Ghosh and Clyde (2011).
  ##     It adds a set of latent observations that make the X'X matrix
  ##     diagonal, simplifying complete data MCMC for model selection.
  ##     ODA is likely to be faster than SSVS, but it is a newer
  ##     method and not quite as battle tested.
  ##   oda.options: A list (which is ignored unless bma.method ==
  ##     "ODA") with the following elements:
  ##     * fallback.probability: Each MCMC iteration will use SSVS
  ##         instead of ODA with this probability.  In cases where the
  ##         latent data have high leverage, ODA mixing can suffer.
  ##         Mixing in a few SSVS steps can help keep an errant
  ##         algorithm on track.
  ##     * eigenvalue.fudge.factor: The latent X's will be chosen so
  ##         that the complete data X'X matrix (after scaling) is a
  ##         constant diagonal matrix equal to the largest eigenvalue
  ##         of the observed (scaled) X'X times (1 +
  ##         eigenvalue.fudge.factor).  This should be a small
  ##         positive number.
  ##   ping: Write a status update to the console every 'ping'
  ##     iterations.
  ##   seed:  An integer to use for the C++ seed.
  ##   ... : Parameters to be passed to SpikeSlabPrior or
  ##     IndependentSpikeSlabPrior., if 'prior' is not specified
  ##     directly.
  ##
  ## Returns:
  ##   An object of class 'lm.spike', which is a list containing the
  ##   following values
  ##   beta: A 'niter' by 'ncol(X)' matrix of regression coefficients
  ##     many of which may be zero.  Each row corresponds to an MCMC
  ##     iteration.
  ##   sigma: A vector of length 'niter' containing the MCMC draws of
  ##     the residual standard deviation parameter.
  ##   prior:  The prior that was used to fit the model.
  ## In addition, the returned object contains sufficient details for
  ## the call to model.matrix in the predict.lm.spike method.

  function.call <- match.call()
  frame <- match.call(expand.dots = FALSE)
  has.data <- !missing(data)
  name.positions <- match(c("formula", "data", "subset", "na.action"),
                          names(frame), 0L)
  frame <- frame[c(1L, name.positions)]
  frame$drop.unused.levels <- drop.unused.levels
  frame[[1L]] <- as.name("model.frame")
  frame <- eval(frame, parent.frame())
  model.terms <- attr(frame, "terms")
  y <- model.response(frame, "numeric")

  x <- model.matrix(model.terms, frame, contrasts)
  bma.method <- match.arg(bma.method)
  error.distribution <- match.arg(error.distribution)
  if (is.null(prior)) {
    if (error.distribution == "gaussian") {
      if (bma.method == "SSVS") {
        prior <- SpikeSlabPrior(x, y, ...)
      } else if (bma.method == "ODA") {
        prior <- IndependentSpikeSlabPrior(x, y, ...)
      }
    } else {
      prior <- StudentSpikeSlabPrior(x, y, ...)
    }
  }
  stopifnot(inherits(prior, "SpikeSlabPriorBase"))
  if (error.distribution == "student") {
    stopifnot(inherits(prior, "StudentSpikeSlabPrior"))
  }
  if (bma.method == "ODA") {
    stopifnot(inherits(prior, "IndependentSpikeSlabPrior"))
    stopifnot(is.list(oda.options))
    check.scalar.probability(oda.options$fallback.probability)
    check.positive.scalar(oda.options$eigenvalue.fudge.factor)
  }

  stopifnot(is.numeric(ping))
  stopifnot(length(ping) == 1)

  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  stopifnot(nrow(x) == length(y))
  stopifnot(length(prior$mu) == ncol(x))
  stopifnot(length(prior$prior.inclusion.probabilities) == ncol(x))

  ## The following is for backward compatibility.  A max.flips
  ## argument was added to SpikeSlabPrior.
  if (is.null(prior$max.flips)) {
    prior$max.flips <- -1
  }

  ans <- .Call(analysis_common_r_do_spike_slab,
               x,
               y,
               prior,
               error.distribution,
               as.integer(niter),
               as.integer(ping),
               bma.method,
               oda.options,
               seed)
  if (!is.null(colnames(x))) {
    colnames(ans$beta) <- colnames(x)
  }

  ## Model diagnostics
  n <- length(y)
  sdy <- sd(y, na.rm = TRUE)
  ans$prior <- prior
  ans$error.distribution <- error.distribution

  ## The stuff below will be needed by predict.lm.spike.
  ans$contrasts <- attr(x, "contrasts")
  ans$xlevels <- .getXlevels(model.terms, frame)
  ans$call <- function.call
  ans$terms <- model.terms
  ans$sample.sd <- sdy
  ans$sample.size <- n

  if (error.distribution == "gaussian") {
    ans$log.likelihood <- -0.5 * n * log(2 * pi) - n * log(ans$sigma) -
        0.5 * ans$sse / ans$sigma^2
    ans$null.log.likelihood <- -0.5 * n * log(2 * pi) - n * log(sdy) -
        0.5 * (n - 1)
  } else if (error.distribution == "student") {
    errors <- matrix(rep(y, times = niter), ncol = niter) - x %*% t(ans$beta)
    errors <- errors / rep(ans$sigma, each = n)
    ans$log.likelihood <-
      colSums(dt(errors,
                 df = rep(ans$tail.thickness, each = n),
                 log = TRUE))
    ## The null log likelihood is not quite right.  We really should
    ## be refitting the model using a regression with no predictors,
    ## but that is potentially expensive, and this will get us in the
    ## ballpark.
    ans$null.log.likelihood <- sum(
        dt(y - mean(y) / sdy,
           df = median(ans$tail.thickness),
           log = TRUE))
  }

  ans$response <- y
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
    ans$training.data <- frame
  }

  ## Methods that work for all glm model families rely on objects
  ## being of class glm.spike.  Unlike the base R stats package,
  ## lm.spike objects also inherit from glm.spike.
  class(ans) <- c("lm.spike", "glm.spike")
  return(ans)
}

##----------------------------------------------------------------------
plot.lm.spike <- function(
    x,
    y = c("inclusion", "coefficients", "scaled.coefficients",
        "residuals", "size", "help"),
    burn = SuggestBurnLogLikelihood(x$log.likelihood),
    ...) {
  ## S3 method for plotting lm.spike objects.
  y <- match.arg(y)
  if (y == "inclusion") {
    return(PlotMarginalInclusionProbabilities(
        x$beta, burn = burn, ...))
  } else if (y == "coefficients") {
    return(PlotLmSpikeCoefficients(
        x$beta, burn = burn, ...))
  } else if (y == "scaled.coefficients") {
    scale.factors = apply(model.matrix(x), 2, sd)
    return(PlotLmSpikeCoefficients(
        x$beta, burn = burn, scale.factors = scale.factors, ...))
  } else if (y == "size") {
    return(PlotModelSize(x$beta, burn = burn, ...))
  } else if (y == "residuals") {
    PlotLmSpikeResiduals(x, burn = burn, ...)
  } else if (y == "help") {
    help("plot.lm.spike", package = "BoomSpikeSlab", help_type = "html")
  } else {
    stop("unknown option", y, " in plot.lm.spike")
  }
}

PlotMarginalInclusionProbabilities <- function(
    beta,
    burn = 0,
    inclusion.threshold = 0,
    unit.scale = TRUE,
    number.of.variables = NULL,
    ...) {
  ## Produces a barplot of the marginal inclusion probabilities for a
  ## set of model coefficients sampled under a spike and slab prior.
  ## The coefficients are sorted by the marginal inclusion
  ## probability, and shaded by the conditional probability that a
  ## coefficient is positive, given that it is nonzero.
  ##
  ## Args:
  ##   beta: A matrix of model coefficients.  Each row represents an
  ##     MCMC draw.  Each column represents a coefficient for a
  ##     variable.
  ##   burn: Number of MCMC iterations to discard.  If burn <= 0 then
  ##     no iterations are discarded.
  ##   inclusion.threshold: Only plot coefficients with posterior
  ##     inclusion probabilites exceeding this value.
  ##   unit.scale: Logical indicating if the axis of the plot should
  ##     be forced to [0, 1].
  ##   number.of.variables: If non-NULL this specifies the number of
  ##     coefficients to plot, taking precedence over
  ##     inclusion.threshold.
  ##   ...:  Additional arguments to be passed to barplot.
  ## Returns:
  ## A list with the following elements:
  ##   barplot: The midpoints of each bar, which is useful for adding
  ##     to the plot.
  ##   inclusion.prob: The marginal inclusion probabilities of each
  ##     variable, ordered smallest to largest (the same ordering as
  ##     the plot).
  ##   positive.prob: The probability that each variable has a
  ##     positive coefficient, in the same order as inclusion.prob.
  ##   permutation: The permutation of beta that puts the coefficients
  ##     in the same order as positive.prob and inclusion.prob.  That
  ##     is: beta[, permutation] will have the most significant
  ##     coefficients in the right hand columns.
  stopifnot(is.matrix(beta))
  stopifnot(nrow(beta) > burn)
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  inclusion.prob <- colMeans(beta != 0)
  index <- order(inclusion.prob)
  beta <- beta[, index, drop = FALSE]
  inclusion.prob <- inclusion.prob[index]

  compute.positive.prob <- function(x) {
    ## Compute the probability that x is positive, given that it is
    ## nonzero.  If all(x == 0) then zero is returned.
    x <- x[x != 0]
    if (length(x) == 0) {
      return(0)
    }
    return(mean(x > 0))
  }
  positive.prob <- apply(beta, 2, compute.positive.prob)

  ## If there are too many variables floating around you won't be able
  ## to read the plot.  You can make a nicer looking plot by only
  ## showing the variables with inclusion probabilities above a
  ## threshold, or by specifying number.of.variables.
  if (!is.null(number.of.variables)) {
    stopifnot(number.of.variables > 0)
    show <- tail(seq(along = inclusion.prob), number.of.variables)
  } else {
    show <- inclusion.prob >= inclusion.threshold
  }

  bar.plot <- NULL
  if (sum(show) > 0) {
    omai <- par("mai")
    variable.names <- colnames(beta)
    omai[2] <- max(strwidth(variable.names[show], units = "inches")) + .5
    oldpar <- par(mai = omai)
    on.exit(par(oldpar))
    xlim <- c(0, max(inclusion.prob[show]))
    if (unit.scale) {
      xlim <- c(0, 1)
    }
    bar.plot <- barplot(inclusion.prob[show],
                        horiz = TRUE,
                        xlim = xlim,
                        col = gray(positive.prob[show]),
                        names.arg = variable.names[show],
                        las = 1,
                        xlab = "Inclusion Probability",
                        ...
                        )
  } else {
    bar.plot <-
        plot(1, 1,
             axes = F, type = "n",
             main = paste("No variables exceeded the inclusion ",
                 "probability threshold of",
                 inclusion.threshold))
  }
  ans <- list(barplot = bar.plot,
              inclusion.prob = inclusion.prob,
              positive.prob = positive.prob,
              permutation = index)
  return(invisible(ans))
}

##----------------------------------------------------------------------
residuals.lm.spike <- function(
    object,
    burn = SuggestBurnLogLikelihood(object$log.likelihood),
    mean.only = FALSE,
    ...) {
  ## Args:
  ##   object:  An lm.spike object.
  ##   burn: Number of MCMC iterations to discard.  If burn <= 0 then
  ##     no iterations are discarded.
  ##   mean.only: Logical.  If TRUE then the posterior mean of each
  ##     residual is returned.  If FALSE then the full posterior
  ##     distribution of residuals is returned.
  ##   ...: not used.  Present only to comply with the signature of
  ##     the residuals() generic function.
  ##
  ## Returns:
  ##   The posterior distribution (or posterior mean) of residuals
  ##   from the model object.  If mean.only is TRUE then the return
  ##   value is the vector of residuals, otherwise the return value is
  ##   a matrix, with rows corresponding to MCMC iterations, and
  ##   columns to individual observations.
  ##
  ## Details:
  ##   This function must be called in the same frame as the one used
  ##   to create the lm.spike model object, and the data objects used
  ##   to fit the model must not have changed between the call and the
  ##   model fit.  This is because lm.spike does not store the model
  ##   matrix as part of the returned object, so it will be
  ##   reconstructed here.
  predictors <- model.matrix(object)
  predictions <- predictors %*% t(object$beta)
  residuals <- t(object$response - predictions)
  if (mean.only) {
    return(colMeans(residuals))
  } else {
    return(residuals)
  }
}

##----------------------------------------------------------------------
PlotLmSpikeCoefficients <- function(beta,
                                    burn = 0,
                                    inclusion.threshold = 0,
                                    scale.factors = NULL,
                                    number.of.variables = NULL,
                                    ...) {
  ## Produces boxplots showing the marginal distribution of the coefficients.
  ##
  ## Args:
  ##   beta: A matrix of model coefficients.  Each row represents an
  ##     MCMC draw.  Each column represents a coefficient for a
  ##     variable.
  ##   burn: Number of MCMC iterations to discard.  If burn <= 0 then
  ##     no iterations are discarded.
  ##   inclusion.threshold: Only plot coefficients with posterior
  ##     inclusion probabilites exceeding this value.
  ##   scale.factors: If non-null then a vector of scale factors with which to
  ##     scale the columns of beta.  A NULL value is ignored.
  ##   number.of.variables: If non-null then this is an upper bound on
  ##     the number of coefficients to show.  Coefficients are ordered
  ##     by their posterior inclusion probabilites.
  ##
  ## Returns:
  ##   The return value of the call to boxplot.
  ##
  ## Details:
  ##   Produces a set of boxplots on the current graphics device.  The
  ##   boxes represent the marginal posterior distribution of the
  ##   regression coefficients beta, ordered by their posterior
  ##   inclusion probabilites.  The widths of the boxes and sizes of
  ##   the points are proportional to the marginal inclusion
  ##   probabilites.
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  if (!is.null(scale.factors)) {
    stopifnot(is.numeric(scale.factors),
              length(scale.factors) == ncol(beta))
    if (any(scale.factors < 0)) {
      warning("Negative scale factors.")
    }
    beta <- t(t(beta) * scale.factors)
  }

  inclusion.prob <- colMeans(beta != 0)
  if (inclusion.threshold < 0) {
    inclusion.threshold <- 0
  }
  included <- inclusion.prob > inclusion.threshold

  omai <- par("mai")
  variable.names <- colnames(beta)
  omai[2] <- max(strwidth(variable.names, units = "inches")) + .5
  oldpar <- par(mai = omai)
  on.exit(par(oldpar))

  beta <- beta[, included]
  inclusion.prob <- inclusion.prob[included]

  if (!is.null(number.of.variables)) {
    stopifnot(is.numeric(number.of.variables),
              length(number.of.variables) == 1,
              number.of.variables > 0)
    if (ncol(beta) > number.of.variables) {
      beta <- beta[, 1:number.of.variables]
      inclusion.prob <- inclusion.prob[1:number.of.variables]
    }
  }

  ord <- order(inclusion.prob)
  beta <- beta[, ord]
  inclusion.prob <- inclusion.prob[ord]

  return(invisible(boxplot(beta,
          width = inclusion.prob,
          horizontal = TRUE,
          las = 1,
          cex = inclusion.prob,
          ...)))
}

##----------------------------------------------------------------------
PlotLmSpikeResiduals <- function(
    object,
    burn = SuggestBurnLogLikelihood(object$log.likelihood),
    ...) {
  ## Args:
  ##   object: An lm.spike model object.
  ##   burn: Number of MCMC iterations to discard.  If burn <= 0 then
  ##     no iterations are discarded.
  ##   ...: Extra arguments passed to 'plot'.
  ##
  ## Returns:
  ##   NULL
  ##
  ## Details:
  ##   Plots residuals vs predicted values, after discarding the
  ##   requested burn-in.  Residuals are posterior mean residuals, and
  ##   predicted values are posterior mean predictions.
  predictors <- model.matrix(object)
  beta <- object$beta
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  predictions <- predictors %*% t(beta)
  residuals <- rowMeans(object$response - predictions)
  predictions <- rowMeans(predictions)
  plot(predictions,
       residuals,
       xlab = "Posterior mean predictions",
       ylab = "Posterior mean residuals",
       ...)
  return(NULL)
}

##----------------------------------------------------------------------
## TODO(stevescott): This code is experimental.  Make a decision to
## keep it or cut it.
## PlotLmSpikeFitSummary <- function(
##     model,
##     burn = SuggestBurnLogLikelihood(model$log.likelihood),
##     cutpoint.basis = c("sample.size", "equal.range"),
##     number.of.buckets = 10,
##     ...) {
##   stopifnot(inherits(model, "lm.spike"))
##   cutpoint.basis <- match.arg(cutpoint.basis)
##   fitted <- predict(model, mean.only = TRUE, burn = burn)
##   if (cutpoint.basis == "sample.size") {
##     cutpoints <- quantile(fitted, (0:number.of.buckets) / number.of.buckets)
##   } else if (cutpoint.basis == "equal.range") {
##     fitted.range <- range(fitted, na.rm = TRUE)
##     cutpoints <- seq(min(fitted.range),
##                      max(fitted.range),
##                      len = number.of.buckets + 1)
##   }
##   cutpoints <- unique(cutpoints)
##   bucket.indicators <- cut(fitted, cutpoints)

##   resids <- residuals(model, mean.only = TRUE, burn = burn)
##   if (is.matrix(resids)) {
##     buckets <- list()
##     unique.buckets <- sort(unique(bucket.indicators))
##     for (bucket in unique.buckets) {
##       buckets[[bucket]] <- unlist(resids[, bucket.indicators == bucket])
##     }
##   } else {
##     buckets <- split(resids, bucket.indicators)
##   }

##   cutpoint.widths <- diff(cutpoints)
##   cutpoint.means <- (head(cutpoints, -1) + tail(cutpoints, -1)) / 2
##   plot(fitted, fitted, type = "n", ylim = range(unlist(buckets), na.rm = TRUE),
##        xlab = "predicted", ylab = "residuals")
##   boxplot(buckets, add = TRUE, at = cutpoint.means,
##           width = cutpoint.widths,
##           axes = FALSE)
##   abline(v = cutpoints, lty = 3)
##   abline(h = 0, lwd = 2)
## }
##----------------------------------------------------------------------
PlotModelSize <- function(
    beta,
    burn = 0,
    xlab = "Number of nonzero coefficients",
    ...) {
  ## Plot a histogram of the number of nonzero coefficient in the model.
  ## Args:
  ##   beta: A matrix of model coefficients.  Each row represents an
  ##     MCMC draw.  Each column represents a coefficient for a
  ##     variable.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   xlab:  Label for the x axis.
  ##   ...:   Extra arguments to be passed to 'hist'.
  ## Returns:
  ##   A vector giving the number of nonzero coefficients in each MCMC
  ##     iteration.
  stopifnot(is.matrix(beta))
  stopifnot(nrow(beta) > burn)
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  size <- rowSums(beta != 0)
  hist(size, xlab = xlab, ...)
  return(invisible(size))
}

##----------------------------------------------------------------------
SummarizeSpikeSlabCoefficients <- function(beta, burn = 0, order = TRUE) {
  ## Compute posterior means, posterior standard deviations, and
  ## posterior inclusion probabilities for the coefficients in the
  ## lm.spike object.
  ## Args:
  ##   beta: a matrix containing MCMC draws of regression coefficients.
  ##     Each row is a draw.  Each column is a coefficient.
  ##   burn:  The number of MCMC draws to be discarded as burn-in.
  ##   order: Logical.  If TRUE the output will be in decending order
  ##     according to the probability that each coefficient is
  ##     nonzero.
  ## Returns:
  ## A five-column matrix giving the posterior mean and standard
  ## deviation of each coefficient, the conditional mean and standard
  ## deviation of each coefficient given that it is nonzero, and the
  ## probability that the coefficient is nonzero.
  beta <- as.matrix(beta)
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
  }
  inclusion.prob <- colMeans(beta != 0)
  if (order) {
    index <- rev(order(inclusion.prob))
    beta <- beta[, index, drop = FALSE]
    inclusion.prob <- inclusion.prob[index]
  }

  ConditionalMean <- function(b) {
    ## Returns the mean of the nonzero elements of b
    b <- b[b != 0]
    if (length(b) > 0) return(mean(b))
    return(0)
  }
  ConditionalSd <- function(b) {
    ## Returns the standard deviation of the nonzero elements of b
    b <- b[b != 0]
    if (length(b) > 1) return(sd(b))
    return(0)
  }

  ## Short names are used here because they are displayed as column
  ## headers.  Names that are wider than the corresponding column of
  ## numbers make the result poorly spaced.
  coef <- cbind(mean = colMeans(beta),
                sd = apply(beta, 2, sd),
                mean.inc = apply(beta, 2, ConditionalMean),
                sd.inc = apply(beta, 2, ConditionalSd),
                inc.prob = inclusion.prob)

  return(coef)
}

##----------------------------------------------------------------------
summary.lm.spike <- function(object, burn = 0, order = TRUE, ...) {
  ## Summarize the coefficients and residual standard deviation from
  ## an lm.spike object.
  ## Args:
  ##   object: An object of class 'lm.spike', or an equivalent
  ##     list-like object containing a matrix of coefficient draws named
  ##     'beta'.
  ##   burn: An integer giving the number of draws to be discarded as
  ##     burn-in.
  ##   order: Logical.  If TRUE then the coefficients are presented in
  ##     order of their posterior inclusion probabilities.  Otherwise
  ##     the coefficients retain the order they had in 'object'.
  ## Returns:
  ## A list with three elements:
  ##   coefficients: A summary of the model coefficients produced by
  ##     SummarizeSpikeSlabCoefficients
  ##   residual.sd: A summary of the posterior distribution of the
  ##     residual standard deviation parameter ("sigma")
  ##   rsquare: A summary of the posterior distribution of the R^2
  ##     statistic: 1 - resdual.sd^2 / sample.sd^2, where sample.sd is
  ##     the sample standard deviation of the response variable.
  sigma <- object$sigma
  if (burn > 0) {
    sigma <- sigma[-(1:burn)]
  }
  rsquare.distribution <- 1 - sigma^2 / object$sample.sd^2
  ans <- list(coefficients =
              SummarizeSpikeSlabCoefficients(object$beta, burn, order),
              residual.sd = summary(sigma),
              rsquare = summary(rsquare.distribution),
              rsquare.distribution = rsquare.distribution)
  class(ans) <- "summary.lm.spike"
  return(ans)
}

##----------------------------------------------------------------------
print.summary.lm.spike <- function(x, ...) {
  ## Print method for summary.lm.spike objects.  Only print 3
  ## significant digits.
  cat("coefficients:\n")
  print.default(signif(x$coefficients, 3), ...)
  cat("\nresidual.sd = \n")
  print(x$residual.sd)
  cat("\nr-square    = \n")
  print(x$rsquare)
  cat("\n")
}

##----------------------------------------------------------------------
predict.lm.spike <- function(object,
                             newdata = NULL,
                             burn = 0,
                             na.action = na.pass,
                             mean.only = FALSE,
                             ...) {
  ## Prediction method for lm.spike
  ## Args:
  ##   object: object of class "lm.spike" returned from the lm.spike
  ##     function
  ##   newdata: Either NULL, or else a data frame, matrix, or vector
  ##     containing the predictors needed to make the prediction.  If
  ##     'newdata' is 'NULL' then the predictors are taken from the
  ##     training data used to create the model object.  Note that
  ##     'object' does not store its training data, so the data
  ##     objects used to fit the model must be present for the
  ##     training data to be recreated.  If 'newdata' is a data.frame
  ##     it must contain variables with the same names as the data
  ##     frame used to fit 'object'.  If it is a matrix, it must have
  ##     the same number of columns as object$beta.  (An intercept
  ##     term will be implicitly added if the number of columns is one
  ##     too small.)  If the dimension of object$beta is 1 or 2, then
  ##     newdata can be a vector.
  ##   burn: The number of MCMC iterations in 'object' that should be
  ##     discarded.  If burn <= 0 then all iterations are kept.
  ##   na.action: what to do about NA's.
  ##   mean.only: Logical.  If TRUE then return the posterior mean of
  ##     the predictive distribution.  If FALSE then return the entire
  ##     distribution.
  ##   ...: extra aguments ultimately passed to model.matrix (in the
  ##     event that newdata is a data frame)
  ## Returns:
  ## A matrix of predictions, with each row corresponding to a row
  ## in newdata, and each column to an MCMC iteration.

  if (is.null(newdata)) {
    predictor.matrix <- model.matrix(object)
  } else {
    predictor.matrix <- GetPredictorMatrix(
        object, newdata, na.action = na.action, ...)
  }
  beta <- object$beta
  sigma <- object$sigma
  nu <- object$tail.thickness
  stopifnot(is.numeric(burn) || length(burn) == 1)
  if (burn > 0) {
    beta <- beta[-(1:burn), , drop = FALSE]
    sigma <- sigma[-(1:burn)]
    if (!is.null(nu)) {
      nu <- nu[-(1:burn)]
    }
  }

  ## eta is the [n x niter] matrix of predicted values
  eta <- predictor.matrix %*% t(beta)
  if (mean.only) {
    return(rowMeans(eta))
  } else {
    if (is.null(nu)) {
      ## Handle Gaussian case.
      y.new <- matrix(rnorm(length(eta),
                            eta,
                            rep(sigma, each = nrow(predictor.matrix))),
                      nrow = nrow(predictor.matrix))
    } else {
      ## Handle student T case.
      y.new <- matrix(rt(length(eta),
                         rep(nu, each = nrow(predictor.matrix)))
                      * rep(sigma, each = nrow(predictor.matrix))
                      + eta,
                      nrow = nrow(predictor.matrix))
    }
    return(y.new)
  }
}
