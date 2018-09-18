HiddenLayer <- function(number.of.nodes, prior = NULL, include.intercept = TRUE) {
  ## The same prior will be used for all nodes.  The dimension of the prior must
  ## match the inputs from the previous layer.
  ##
  ## Args:
  ##   number.of.nodes: The number of output nodes in this hidden layer.  The
  ##     number of inputs is determined by the preceding layer.
  ##   prior: An MvnPrior or SpikeSlabGlmPrior to use for the coefficients of
  ##     each of the logistic regression models comprising this hidden layer.
  ##     All models will use the same prior.
  ##   include.intercept: Logical value indicating whether the first output for
  ##     this layer should be forced "on" with probability 1, so that it can act
  ##     as an intercept for the next layer.
  
  check.scalar.integer(number.of.nodes)
  stopifnot(number.of.nodes > 0)
  stopifnot(inherits(prior, "MvnPrior") ||
              inherits(prior, "SpikeSlabGlmPrior"))
  check.scalar.boolean(include.intercept)
  ans <- list(number.of.nodes = number.of.nodes,
    prior = prior,
    include.intercept)
  class(ans) <- c("HiddenLayerSpecification")
  return(ans)
}

.CheckHiddenLayers <- function(predictors, hidden.layers) {
  ## Check that the priors for each hidden layer are set and are of the
  ## appropriate dimension.  Replace any NULL priors with default values.
  ## Return the updated list of hidden layers.
  stopifnot(is.list(hidden.layers),
    all(sapply(hidden.layers, inherits, "HiddenLayerSpecification")))

  for (i in 1:length(hidden.layers)) {
    if (is.null(hidden.layers[[i]]$prior)) {
      if (i == 1) {
        ## Logistic regression inputs are arbitrary.
        prior <- LogitZellnerPrior(predictors,
          prior.success.probability = .5,
          expected.model.size = 5)
        hidden.layers[[i]]$prior <- prior
      } else {
        ## Logistic regression inputs are all between 0 and 1.
        input.dimension <- hidden.layers[[i - 1]]$number.of.nodes
        prior <- SpikeSlabGlmPriorDirect(
          coefficient.mean = rep(0, input.dimension),
          coefficient.precision = diag(rep(1, input.dimension)),
          prior.inclusion.probabilities =
            rep(5 / input.dimension, input.dimension))
        hidden.layers[[i]]$prior <- prior
      }
    }

    stopifnot(inherits(hidden.layers[[i]]$prior, "SpikeSlabPriorBase"))
    if (i == 1) {
      stopifnot(length(hidden.layers[[i]]$prior$mu) == ncol(predictors))
    } else {
      stopifnot(length(hidden.layers[[i]]$prior$mu) ==
                  hidden.layers[[i - 1]]$number.of.nodes)
    }
  }
  return(hidden.layers)
}

.EnsureTerminalLayerPrior <- function(response, hidden.layers, prior, ...) {
  ## Args:
  ##   response:  The vector of 'y' values from the regression.
  ##   hidden.layers: A list of objects inheriting from
  ##     HiddenLayerSpecification.
  ##   prior: The prior distribution for the model in the terminal layer.  This
  ##     must be of type SpikeSlabPrior, SpikeSlabPriorDirect, or NULL.  If NULL
  ##     a default prior will be created.
  ##
  ## Returns:
  ##   The checked prior distribution.
  ##
  ## Effects:
  ## Checks that the dimension of the prior matches the number of outputs in the
  ## final hidden layer.  
  dimension <- tail(hidden.layers)[[1]]$number.of.nodes
  if (is.null(prior)) {
    precision <- diag(rep(1, dimension))
    if (dimension > 5) {
      inclusion.probabilities <- rep(5.0 / dimension, dimension)
    } else {
      inclusion.probabilities <- rep(.5, dimension)
    }
    prior <- SpikeSlabPriorDirect(
      coefficient.mean = rep(0, dimension),
      coefficient.precision = precision,
      prior.inclusion.probabilities = inclusion.probabilities,
      prior.sigma.guess = sd(response, na.rm = TRUE) / 2,
      prior.sigma.sample.size = 1)
  }
  stopifnot(inherits(prior, "SpikeSlabPrior")
    || inherits(prior, "SpikeSlabPriorDirect"))
  return(prior)
}

BayesNnet <- function(formula,
                      hidden.layers,
                      niter,
                      data,
                      subset,
                      prior = NULL,
                      drop.unused.levels = TRUE,
                      ping = niter / 10,
                      seed = NULL,
                      ...) {
  ## Args:
  ##   formula:  A model formula as one would pass to 'lm'.
  ##   hidden.layers: A list of HiddenLayer objects defining the network
  ##     structure.
  ##   niter: The desired number of MCMC iterations.
  ##   data: An optional data frame containing the variables used in 'formula'.
  ##   subset:  See 'lm'.
  ##   prior: An object of class SpikeSlabPrior defining the prior distribution
  ##     for the terminal layer.  This includes the prior for the residual
  ##     variance.
  ##   drop.unused.levels:  See 'lm'.
  ##   ping: The frequency with which to print status updates for the MCMC
  ##     algorithm.  Setting 'ping = 10' will print a status update message
  ##     every 10 MCMC iterations.
  ##   seed:  The seed to use for the C++ random number generator.
  ##   ...: Extra arguments are passed to 'SpikeSlabPrior' in the event that
  ##     'prior' is NULL.
  ##
  ## Returns:
  ##   The MCMC draws for all the network coefficients.  The return value also
  ##   includes information needed by supporting methods (e.g. 'plot' and
  ##   'predict').
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
  response <- model.response(frame, "numeric")
  predictors <- model.matrix(model.terms, frame, contrasts)

  ## Check that each layer coheres with the preceding layer.
  hidden.layers <- .CheckHiddenLayers(predictors, hidden.layers)  
  prior <- .EnsureTerminalLayerPrior(response, prior, hidden.layers)  

  check.positive.scalar(niter)
  check.positive.scalar(ping)
  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  ans <- .Call(analysis_common_r_do_feedforward,
    predictors,
    response,
    hidden.layers,
    prior,
    as.integer(niter),
    as.integer(ping),
    seed)

  class(ans) <- c("BayesNnet")
  return(ans)
}

## Plot predicted vs. actual.
## partial dependence plot.
## residuals.
## Network structure: nodes and arrows, with heavier arrows having higher
## inclusion probabilities.
plot.BayesNnet <- function(x, y = c("predicted", "residual", "structure"), ...) {}

PartialDependencePlot <- function(model, variable.names) {}

print.BayesNnet <- function(x, ...) {}

predict.BayesNnet <- function(object, ...) {}
