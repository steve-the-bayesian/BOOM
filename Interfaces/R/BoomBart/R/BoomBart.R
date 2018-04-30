BoomBart <- function(formula, niter, data,
                     family = c("gaussian", "probit", "logit", "poisson"),
                     initial.number.of.trees = 200,
                     tree.prior = NULL,
                     discrete.distribution.limit = 20,
                     continuous.distribution.strategy = c(
                       "uniform.continuous",
                       "uniform.discrete"),
                     ping = niter / 10,
                     seed = NULL,
                     total.prediction.sd = NULL,
                     number.of.trees.prior = DiscreteUniformPrior(1, 200),
                     ...) {
  ## Fit a Bart model using the BOOM Bart implementation.
  ##
  ## Args:
  ##   formula: A model formula such as you might give to lm, glm,
  ##     etc.  The formula should only consist of linear terms as the
  ##     model will figure out any interactions or nonlinearities on
  ##     its own.
  ##   niter:  The number of MCMC iterations to run.
  ##   data: An optional data.frame containing the variables used in
  ##    'formula.'
  ##   family: A string indicating the family of the error
  ##     distribution.
  ##   initial.number.of.trees: The initial number of trees to use in
  ##     the Bart model.  This will vary over the life of the MCMC,
  ##     unless the tree.prior contains a PointMassPrior on the number
  ##     of trees.
  ##   tree.prior: An object of class BartTreePrior specifying the
  ##     prior distribution on the tree topology, the number of trees,
  ##     and the mean parameters at the leaves.
  ##   sigma.prior: An object of class SdPrior for the residual
  ##     variance.  This is only used if family is "gaussian".  If it
  ##     is NULL, then an SdPrior will be constructed based on the
  ##     arguments 'sigma.weight' and 'expected.r2', where the guess
  ##     at sigma is sqrt(1 - expected.r2) * sd(y).
  ##   ping: The frequency with which to print status update messages
  ##     during the MCMC.  For example, ping = 10 will print a message
  ##     every 10 iterations.
  ##   seed: The random seed to use for the BOOM random number
  ##     generator.  An integer.
  ##   number.of.trees.prior: If tree.prior is NULL, this is passed to
  ##     either BartTreePrior or GaussianBartTreePrior.
  ##   ...: Extra arguments passed to either BartTreePrior or
  ##     GaussianBartTreePrior.
  ##
  ## Returns:
  ##   An object of class BoomBart, which is a list containing Monte
  ##   Carlo draws of the trees (represented as lists of matrices),
  ##   and (if family == "gaussian") the residual standard deviation
  ##   "sigma".  Most of the output is not designed for users to
  ##   examine directly, but rather to be used in plot, predict, and
  ##   related methods.
  stopifnot(is.numeric(niter))
  stopifnot(is.numeric(initial.number.of.trees))
  stopifnot(length(initial.number.of.trees) == 1)

  function.call <- match.call()
  my.model.frame <- match.call(expand.dots = FALSE)
  frame.match <- match(c("formula", "data", "na.action"),
                       names(my.model.frame), 0L)
  my.model.frame <- my.model.frame[c(1L, frame.match)]
  my.model.frame$drop.unused.levels <- TRUE

  my.model.frame[[1L]] <- as.name("model.frame")
  my.model.frame <- eval(my.model.frame, parent.frame())
  model.terms <- attr(my.model.frame, "terms")
  ## Never include an intercept term.
  attributes(model.terms)$intercept <- 0

  # Determine the family of the error distribution.
  family <- match.arg(family)

  if (family == "gaussian") {
    y <- model.response(my.model.frame, "numeric")
    y <- as.double(y)
    if (is.null(tree.prior)) {
      sdy <- sqrt(var(y, na.rm = TRUE))
      if (is.null(total.prediction.sd)) {
        total.prediction.sd <- sdy
      }
      tree.prior <- GaussianBartTreePrior(
          total.prediction.sd = total.prediction.sd,
          sdy = sdy,
          number.of.trees.prior = number.of.trees.prior,
          ...)
    }
    stopifnot(inherits(tree.prior, "GaussianBartTreePrior"))
  } else if (family == "probit" || family == "logit") {
    y <- model.response(my.model.frame, "any")
    if (!is.null(dim(y)) && length(dim(y)) > 1) {
      stopifnot(length(dim(y)) == 2, ncol(y) == 2)
      ## If the user passed a formula like "cbind(successes, failures) ~
      ## x", then y will be a two column matrix
      ny <- y[, 1] + y[, 2]
      y <- y[, 1]
    } else {
      ## The following line admits y's which are TRUE/FALSE, 0/1 or 1/-1.
      y <- y > 0
      ny <- rep(1, length(y))
    }
    y <- cbind(y, ny)
    if (is.null(total.prediction.sd)) {
      total.prediction.sd <- ifelse(family == "probit", 1.0, 1.8)
    }
  } else if (family == "poisson") {
    y <- model.response(my.model.frame, "numeric")
    y <- as.integer(y)
    if (is.null(total.prediction.sd)) {
      # For Poisson data the variance is is equal to the mean.  The
      # log maps the variation to the log scale to match the log link
      # function.
      total.prediction.sd <- .5 * log(mean(y))
    }
  }

  if (is.null(tree.prior)) {
    # Will only get here if family != "gaussian", because the default
    # prior will have already been assigned in the Gaussian case.
    stopifnot(is.numeric(total.prediction.sd)
              && length(total.prediction.sd) == 1
              && total.prediction.sd > 0)
    tree.prior <- BartTreePrior(total.prediction.sd,
                                number.of.trees.prior = number.of.trees.prior,
                                ...)
  }

  stopifnot(inherits(tree.prior, "BartTreePrior"))

  design.matrix <- model.matrix(model.terms, my.model.frame, contrasts)
  stopifnot(nrow(design.matrix) == length(y))

  discrete.distribution.limit <- as.integer(discrete.distribution.limit)
  stopifnot(length(discrete.distribution.limit) == 1)

  continuous.distribution.strategy <-
    match.arg(continuous.distribution.strategy)

  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  ans <- .Call("boom_bart_wrapper_",
               as.integer(initial.number.of.trees),
               design.matrix,
               y,
               family,
               tree.prior,
               discrete.distribution.limit,
               continuous.distribution.strategy,
               as.integer(niter),
               as.integer(ping),
               seed,
               PACKAGE = "BoomBart")
  class(ans) <- "BoomBart"

  colnames(ans$tree.size.distribution) <-
    c("trees", "min", ".10", ".25", ".50", ".75", ".90", "max")

  ans$terms <- model.terms
  ans$contrasts <- attr(design.matrix, "contrasts")
  ans$family <- family
  ans$prior <- tree.prior
  ans$design.matrix <- design.matrix
  ans$response <- y

  return(ans)
}

##======================================================================
predict.BoomBart <- function(object,
                             newdata,
                             distribution = c("function", "prediction"),
                             burn = SuggestBartBurn(object),
                             thin = 10,
                             scale = c("trees", "data"),
                             mean.only = FALSE,
                             ...) {
  ## S3 method for making predictions based on a BoomBart model.
  ## Args:
  ##   object:  A BoomBart model on which to base the predictions.
  ##   newdata: A data.frame containing the variables used in
  ##     object$formula.
  ##   distribution: A string indicating the type of posterior
  ##     predictive distribution desired.  If "function" then the
  ##     predictive distribution of function values at 'newdata' is
  ##     returned.  If "prediction" then the posterior predictive
  ##     distribution of a new data value is returned.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   thin: The frequency of MCMC iterations to keep, after burn-in.
  ##     E.g. if thin = 10 then every 10th draw will be used.
  ##   scale: The scale on which the predictions are to be made.  If
  ##     this is is "trees" then they predicted values are on the
  ##     link-function scale.  E.g. on the logit or probit scale for
  ##     binary data, or the log scale for Poisson data.  If 'scale'
  ##     is "data" then the inverse link function is applied.
  ##   ...: Extra arguments are not used.  This argument is here to
  ##     comply with the signatrue of the default S3 predict method.
  ##
  ## Returns:
  ##   A list with the following elements.
  ##   predictive.distribution: A matrix of predictions.  Each row is
  ##     an MCMC draw.  Each column corresponds to a row in 'newdata'.
  ##   prediction: The posterior median of the predictive
  ##     distribution.  This is a vector of length equal to the number
  ##     of rows in 'newdata'.

  stopifnot(inherits(object, "BoomBart"))
  distribution <- match.arg(distribution)
  stopifnot(is.numeric(burn))
  stopifnot(length(burn) == 1)
  stopifnot(burn >= 0)
  newdata <- as.data.frame(newdata)

  tt <- terms(object)
  Terms <- delete.response(tt)
  m <- model.frame(Terms, newdata, xlev = object$xlevels)
  if (!is.null(cl <- attr(Terms, "dataClasses")))
    .checkMFClasses(cl, m)
  X <- model.matrix(Terms, m, contrasts.arg = object$contrasts)
  if (nrow(X) != nrow(newdata)) {
    msg <- paste("Some entries in newdata have missing values, and  will",
                 "be omitted from the prediction.")
    warning(msg)
  }

  predictions <- .Call("boom_bart_prediction_wrapper_",
                       object,
                       X,
                       as.integer(burn),
                       as.integer(thin),
                       PACKAGE = "BoomBart")
  family <- object$family
  if (family == "gaussian" && distribution == "prediction") {
    ## To sample from the posterior predictive distribution, take the
    ## function draws that are currently in 'predictions' and add noise.
    sigma <- object$sigma
    if (burn > 0) {
      sigma <- sigma[-(1:burn)]
    }
    ## Sigma will be repeated for each column of the draw.  Because
    ## rows represent MCMC iterations, repeatedly cycling through
    ## sigma is the right thing to do.
    noise <- matrix(rnorm(length(predictions), 0, sigma),
                    nrow = nrow(predictions))
    predictions <- predictions + noise
  }

  if (family != "gaussian" && scale == "data") {
    inverse.link <- c("poisson" = exp,
                      "probit" = pnorm,
                      "logit" = plogis)[family]
    predictions <- inverse.link(predictions);
  }

  if (mean.only) {
    predictions <- as.numeric(colMeans(predictions))
  }

  class(predictions) <- "BoomBartPrediction"
  return(predictions)
}

##======================================================================
SuggestBartBurn <- function(model, proportion = .1) {
  ## A suggestion as to the number of burn-in iterations to use.
  niter <- length(model$trees)
  return(floor(proportion * niter))
}

##======================================================================
plot.BoomBart <- function(x, y, ...) {
  ## Args:
  ##   x:  A BoomBart model object.
  ##   y: If y is present, it should either be the name of the
  ##     variable for which a partial dependence plot is desired, or
  ##     its index (column number) in the design matrix.  If y is
  ##     missing an overall model summary will be plotted instead.
  ##
  ## Returns:
  ##   The result of the dispatched plotting function.
  if (missing(y)) {
    PlotTreeSizeDistribution(x, ...)
  } else {
    BartPartialDependencePlot(x, which.variable = y, ...)
  }
}
