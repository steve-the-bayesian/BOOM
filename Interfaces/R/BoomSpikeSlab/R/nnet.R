HiddenLayer <- function(number.of.nodes, prior = NULL,
                        expected.model.size = Inf) {
  ## Specify the structure of a hidden layer to be used in a feedforward neural
  ## network.
  ##
  ## TODO(steve): Include an option for a 'bias' term.
  ##
  ## Args:
  ##   number.of.nodes: The number of output nodes in this hidden layer.  The
  ##     number of inputs is determined by the preceding layer.
  ##   prior: An MvnPrior or SpikeSlabGlmPrior to use for the coefficients of
  ##     each of the logistic regression models comprising this hidden layer.
  ##     All models will use the same prior.  The dimension of the prior must
  ##     match the inputs from the previous layer.
  ##
  ## Returns:
  ##   An object (list) encoding the necessary information for the underlying
  ##   C++ code to build the desired neural network model.
  check.scalar.integer(number.of.nodes)
  stopifnot(number.of.nodes > 0)
  stopifnot(is.null(prior) ||
              inherits(prior, "MvnPrior") ||
              inherits(prior, "SpikeSlabGlmPrior") ||
              inherits(prior, "SpikeSlabGlmPriorDirect"))
  ans <- list(number.of.nodes = number.of.nodes,
    prior = prior,
    expected.model.size = expected.model.size)
  class(ans) <- c("HiddenLayerSpecification")
  return(ans)
}
##===========================================================================
BayesNnet <- function(formula,
                      hidden.layers,
                      niter,
                      data,
                      subset,
                      prior = NULL,
                      expected.model.size = Inf,
                      drop.unused.levels = TRUE,
                      contrasts = NULL,
                      ping = niter / 10,
                      seed = NULL) {
  ## A Bayesian feed-forward neural network with logistic activation function
  ## and a Gaussian terminal layer.
  ##
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
  ##   contrasts: An optional list. See the 'contrasts.arg' of
  ##     ‘model.matrix.default’.
  ##   ping: The frequency with which to print status updates for the MCMC
  ##     algorithm.  Setting 'ping = 10' will print a status update message
  ##     every 10 MCMC iterations.
  ##   seed:  The seed to use for the C++ random number generator.
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
  prior <- .EnsureTerminalLayerPrior(
    response = response,
    prior = prior,
    hidden.layers = hidden.layers,
    expected.model.size = expected.model.size)  

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

  ans$hidden.layer.specification <- hidden.layers
  ans$niter <- niter
  ans$contrasts <- attr(predictors, "contrasts")
  ans$xlevels <- .getXlevels(model.terms, frame)
  ans$call <- function.call
  ans$terms <- model.terms
  ans$response <- response
  if (has.data) {
    ans$training.data <- data
  } else {
    ans$training.data <- frame
  }

  dimnames(ans$hidden.layer.coefficients[[1]]) <- 
    list(NULL, colnames(predictors), NULL)

  class(ans) <- c("BayesNnet")
  return(ans)
}

##===========================================================================
plot.BayesNnet <- function(x, y = c("predicted", "residual", "structure",
  "partial", "help"), ...) {
  ## Match the 'y' argument against the supplied default values, or the data frame
  stopifnot(is.character(y))
  which.function <- try(match.arg(y), silent = TRUE)
  which.variable <- "all"
  if (inherits(which.function, "try-error")) {
    which.function <- "partial"
    which.variable <- pmatch(y, names(x$training.data))
    if (is.na(which.variable)) {
      err <- paste0("The 'y' argument ", y, " must either match one of the plot types",
        " or one of the predictor variable names.")
      stop(err)
    }
  } 
  if (which.function == "predicted") {
    PlotBayesNnetPredictions(x, ...)
  } else if (which.function == "residual") {
    PlotBayesNnetResiduals(x, ...)
  } else if (which.function == "structure") {
    PlotNetworkStructure(x, ...)
  } else if (which.function == "help") {
    help("plot.BayesNnet", package = "BoomSpikeSlab", help_type = "html")
  } else if (which.function == "partial") {
    if (which.variable == "all") {
      varnames <- colnames(attributes(x$terms)$factors)
      nvars <- length(varnames)
      stopifnot(nvars >= 1)
      nr <- max(1, floor(sqrt(nvars)))
      nc <- ceiling(nvars / nr)
      original.pars <- par(mfrow = c(nr, nc))
      on.exit(par(original.pars))
      for (i in 1:nvars) {
        PartialDependencePlot(x, varnames[i], xlab = varnames[i], ...)
      }
    } else {
      PartialDependencePlot(x, y, ...)
    }
  }
  return(invisible(NULL))
}
##===========================================================================
SuggestBurn <- function(model) {
  return(SuggestBurnLogLikelihood(-1 * model$residual.sd))
}

##===========================================================================
PlotBayesNnetPredictions <- function(model, burn = SuggestBurn(model), ...) {
  pred <- predict(model, burn = burn)
  predicted <- colMeans(pred)
  actual <- model$response
  plot(predicted, actual, ...)
  abline(a = 0, b = 1)
}
##===========================================================================
PlotBayesNnetResiduals <- function(model, burn = SuggestBurn(model), ...) {
  pred <- predict(model, burn = burn)
  predicted <- colMeans(pred)
  actual <- model$response
  residual <- actual - predicted
  plot(predicted, residual, ...)
  abline(h = 0)
}
##===========================================================================
.HiddenLayerPriorMean <- function(prior) {
  ## Extract the mean of the prior distribution for hidden layer coefficients.
  if (inherits(prior, "SpikeSlabPriorBase")) {
    return(prior$mu)
  } else if (inherits(prior, "MvnPrior")) {
    return(prior$mean)
  } else {
    stop("Prior must be an MvnPrior or else inherit from SpikeSlabPriorBase")
  }
}
##===========================================================================
.CheckHiddenLayers <- function(predictors, hidden.layers) {
  ## Check that the priors for each hidden layer are set and are of the
  ## appropriate dimension.  Replace any NULL priors with default values.
  ## Return the updated list of hidden layers.
  ##
  ## Args:
  ##   predictors:  The matrix of predictors.
  ##   hidden.layers: A list of HiddenLayerSpecification objects defining the
  ##    hidden layers for the feed forward neural network model.
  ##
  ## Returns:
  ##   hidden.layers, after checking that dimensions and priors are okay, and
  ##   after replacing NULL priors with hopefully sensible defaults.
  stopifnot(is.list(hidden.layers),
    all(sapply(hidden.layers, inherits, "HiddenLayerSpecification")))
  for (i in 1:length(hidden.layers)) {
    if (is.null(hidden.layers[[i]]$prior)) {
      expected.model.size <- hidden.layers[[i]]$expected.model.size
      if (is.null(expected.model.size)) {
        expected.model.size <-  10^10
      }
      check.positive.scalar(expected.model.size)

      if (i == 1) {
        ## Logistic regression inputs are arbitrary.
        prior <- LogitZellnerPrior(predictors,
          prior.success.probability = .5,
          expected.model.size = expected.model.size)
        hidden.layers[[i]]$prior <- prior
      } else {
        ## Logistic regression inputs are all between 0 and 1.
        input.dimension <- hidden.layers[[i - 1]]$number.of.nodes
        prior <- SpikeSlabGlmPriorDirect(
          coefficient.mean = rep(0, input.dimension),
          coefficient.precision = diag(rep(1, input.dimension)),
          expected.model.size = expected.model.size)
        hidden.layers[[i]]$prior <- prior
      }
    }
    if (i == 1) {
      stopifnot(length(.HiddenLayerPriorMean(hidden.layers[[i]]$prior))
        == ncol(predictors))
    } else {
      stopifnot(length(.HiddenLayerPriorMean(hidden.layers[[i]]$prior))
        == hidden.layers[[i - 1]]$number.of.nodes)
    }
  }
  return(hidden.layers)
}
##===========================================================================
.EnsureTerminalLayerPrior <- function(response, hidden.layers, prior,
                                      expected.model.size, ...) {
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
  dimension <- tail(hidden.layers, 1)[[1]]$number.of.nodes
  if (is.null(prior)) {
    precision <- diag(rep(1, dimension))
    inclusion.probabilities <- rep(expected.model.size / dimension, dimension)
    inclusion.probabilities[inclusion.probabilities > 1] <- 1
    inclusion.probabilities[inclusion.probabilities < 0] <- 0
    
    prior <- SpikeSlabPriorDirect(
      coefficient.mean = rep(0, dimension),
      coefficient.precision = precision,
      prior.inclusion.probabilities = inclusion.probabilities,
      sigma.guess = sd(response, na.rm = TRUE) / 2,
      prior.df = 1)
  }
  stopifnot(inherits(prior, "SpikeSlabPrior")
    || inherits(prior, "SpikeSlabPriorDirect"))
  return(prior)
}
##===========================================================================
predict.BayesNnet <- function(object, newdata = NULL, burn = 0, na.action = na.pass,
                              mean.only = FALSE, seed = NULL, ...) {
  ## Prediction method for BayesNnet.
  ## Args:
  ##   object: object of class "BayesNnet" returned from the BayesNnet function.
  ##   newdata: Either NULL, or else a data frame, matrix, or vector containing
  ##     the predictors needed to make the prediction.  If 'newdata' is 'NULL'
  ##     then the predictors are taken from the training data used to create the
  ##     model object.  Note that 'object' does not store its training data, so
  ##     the data objects used to fit the model must be present for the training
  ##     data to be recreated.  If 'newdata' is a data.frame it must contain
  ##     variables with the same names as the data frame used to fit 'object'.
  ##     If it is a matrix, it must have the same number of columns as
  ##     object$beta.  (An intercept term will be implicitly added if the number
  ##     of columns is one too small.)  If the dimension of object$beta is 1 or
  ##     2, then newdata can be a vector.
  ##   burn: The number of MCMC iterations in 'object' that should be discarded.
  ##     If burn <= 0 then all iterations are kept.
  ##   na.action: what to do about NA's.
  ##   mean.only: Logical.  If TRUE then return the posterior mean of the
  ##     predictive distribution.  If FALSE then return the entire distribution.
  ##   seed:  Seed for the C++ random number generator.
  ##   ...: extra aguments ultimately passed to model.matrix (in the event that
  ##     newdata is a data frame)
  ## Returns:
  ##   A matrix of predictions, with each row corresponding to a row in newdata,
  ##   and each column to an MCMC iteration.
  if (is.null(newdata)) {
    predictor.matrix <- model.matrix(object, data = object$training.data)
  } else {
    predictor.matrix <- model.matrix(object, data = newdata)
  }
  stopifnot(is.matrix(predictor.matrix),
    nrow(predictor.matrix) > 0)
  check.nonnegative.scalar(burn)
  check.scalar.boolean(mean.only)
  if (!is.null(seed)) {
    check.scalar.integer(seed)
  }
  ans <- .Call(analysis_common_r_feedforward_prediction,
    object,
    predictor.matrix,
    as.integer(burn),
    as.logical(mean.only),
    seed)
  class(ans) <- "BayesNnetPrediction"
  return(ans)
}
##===========================================================================
PlotNetworkStructure <- function(model, ...) {
  ## Plot the nodes and edges of the neural network.  Larger coefficients are
  ## thicker lines.
  ##
  ## Args:
  ##   model:  A model fit by BayesNnet.
  ##   ...: Extra arguments passed to plot.igraph.
  input.names <- dimnames(model$hidden.layer.coefficients[[1]])[[2]]
  input.dimension <- length(input.names)
  input.nodes <- data.frame(id = input.names,
    layer = rep(0, length(input.names)),
    position.in.layer = 1:length(input.names))

  number.of.hidden.layers <- length(model$hidden.layer.specification)
  hidden.node.counts <- sapply(model$hidden.layer.specification,
    function(x) x$number.of.nodes)

  ## Edge weights are the absolute values of the coefficients in each layer,
  ## normalized by that layer so that input and output nodes don't dominate
  ## because of scaling issues.

  layer <- rep(1:number.of.hidden.layers, times = hidden.node.counts)
  position.in.layer <- c(sapply(hidden.node.counts, function(x) 1:x))
  
  hidden.nodes <- data.frame(
    id = paste("H", layer, position.in.layer, sep = "."),
    layer = rep(1:number.of.hidden.layers, times = hidden.node.counts),
    position.in.layer = position.in.layer)

  terminal.node <- data.frame(
    id = "terminal",
    layer = length(hidden.node.counts) + 1,
    position.in.layer = 1)

  nodes <- rbind(input.nodes, hidden.nodes, terminal.node)

  ##---------------------------------------------------------------------------
  ## Compute the edges.
  first.hidden.layer <- hidden.nodes[hidden.nodes$layer == 1, , drop = FALSE]
  weights <- as.numeric(t(colMeans(model$hidden.layer.coefficients[[1]])))
  nc <- max(abs(weights))
  if (nc > 0) weights <- weights / nc
  input.layer.edges <- data.frame(
    from = as.character(rep(input.names, each = hidden.node.counts[1])),
    to = as.character(rep(first.hidden.layer$id, times = length(input.names))),
    weight = weights
  )
  edges <- input.layer.edges

  if (number.of.hidden.layers >= 2) {
    for (layer in 2:number.of.hidden.layers) {
      current.layer <- nodes[nodes$layer == layer, , drop = FALSE]
      previous.layer <- nodes[nodes$layer == layer - 1, , drop = FALSE]
      weights <- as.numeric(t(colMeans(model$hidden.layer.coefficients[[layer]])))
      nc <- max(abs(weights))
      if (nc > 0) weights <- weights / nc
      edges <- rbind(edges, data.frame(
        from = as.character(rep(previous.layer$id, each = hidden.node.counts[layer])),
        to = as.character(rep(current.layer$id, times = nrow(previous.layer))),
        weight = weights
      ))
    }
  }
  final.hidden.layer <-
    nodes[nodes$layer == number.of.hidden.layers, , drop = FALSE]

  weights <- colMeans(model$terminal.layer.coefficients)
  nc <- max(abs(weights))
  if (nc > 0) weights <- weights / nc
  edges <- rbind(edges, data.frame(
    from = as.character(final.hidden.layer$id),
    to = as.character(terminal.node$id),
    weight = weights
  ))
  
  ##---------------------------------------------------------------------------
  ## Compute the layout for the plot.
  max.nodes <- max(nrow(input.nodes), hidden.node.counts)
  initial.layer.node.offset <- (max.nodes - nrow(input.nodes)) / 2
  hidden.layer.node.offsets <- (max.nodes - hidden.node.counts) / 2
  terminal.layer.offset <- (max.nodes - 1) / 2

  input.layer.layout <- cbind("layer" = 0,
  "position.in.layer" = (1:nrow(input.nodes)) + initial.layer.node.offset)
  hidden.layer.layout <- cbind(hidden.nodes[, c("layer", "position.in.layer")])
  hidden.layer.layout[,2] <- hidden.layer.layout[, 2] +
    hidden.layer.node.offsets[hidden.layer.layout[, 1]]
  terminal.layer.layout <- cbind("layer" = 1 + number.of.hidden.layers,
    "position.in.layer" = 1 + terminal.layer.offset)
  graph.layout <-  rbind(input.layer.layout, hidden.layer.layout,
    terminal.layer.layout)
  
  ##---------------------------------------------------------------------------
  ## Do the plotting.
  graph <- graph_from_data_frame(edges, vertices = NULL)
  plot(graph, layout = as.matrix(graph.layout),
    edge.color = edges$weight > 0,
    edge.width = 5 * abs(edges$weight),
    edge.arrow.size = 0,
    ...)
  
  return(invisible(list(nodes = nodes, input.layer.edges = input.layer.edges)))
}
