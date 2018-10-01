PartialDependencePlot <- function(model,
                                  which.variable,
                                  burn = SuggestBurn(model),
                                  data.fraction = .2,
                                  gridsize = 50,
                                  mean.only = FALSE,
                                  show.points = TRUE,
                                  xlab = NULL,
                                  ylab = NULL,
                                  ylim = NULL,
                                  report.time = FALSE,
                                  ...) {
  ## Args:
  ##   model: A model with a suitably defined 'predict' method.  See below for
  ##     requirements.
  ##   which.variable: Either the name of the variable for which a
  ##     partial dependence plot is desired, or its index (column
  ##     number) in the predictor matrix.
  ##   burn: The number of initial MCMC iterations to discard as
  ##     burn-in.
  ##   data.fraction: The fraction of observations in the predictor
  ##     matrix to use when constructing the partial dependence plot.
  ##     A random sub-sample of this fraction will be taken (without
  ##     replacement).
  ##   mean.only: Logical.  If TRUE then only the mean is plotted at
  ##     each point.  If FALSE then the posterior of the function
  ##     value is plotted.
  ##   show.points: If TRUE then the scatterplot of x vs y is added to
  ##     the graph.  Otherwise the points are left off.
  ##   xlab: Label for the X axis.  NULL produces a default label.
  ##     Use "" for no label.
  ##   ylab: Label for the Y axis.  NULL produces a default label.
  ##     Use "" for no label.
  ##   ylim: Limits on the vertical axis.  If NULL then the plot will
  ##     default to its natural vertical limits.
  ##   ...: Extra arguments are passed either to 'plot' (if mean.only
  ##     is TRUE)' or 'PlotDynamicDistribution' (otherwise).
  ##
  ## Returns:
  ##   Invisibly returns a list containing the posterior of the
  ##   distribution function values at each 'x' value, and the x's
  ##   where the distribution is calculated.
  stopifnot(inherits(model, "BayesNnet"))
  training.data <- model$training.data
  check.scalar.probability(data.fraction)
  if (data.fraction < 1) {
    rows <- sample(1:nrow(training.data),
      size = round(data.fraction * nrow(training.data)))
    training.data <- training.data[rows, , drop = FALSE]
  }

  if (report.time) {
    start.time <- Sys.time()
    cat("Starting at\t", format(start.time), "\n")
  }

  ## which.variable can either be a number or a name.  If it is a name, convert
  ## it to a number.
  stopifnot(length(which.variable) == 1)
  if (is.character(which.variable)) {
    original.variable.name <- which.variable
    which.variable <- pmatch(which.variable, colnames(training.data))
    if (is.na(which.variable)) {
      stop(original.variable.name, "was not uniquely matched in",
        colnames(model$training.data))
    }
  } else {
    which.variable <- as.numeric(which.variable)
    original.variable.name <- colnames(training.data)[which.variable]
  }

  ## The training data defines a marginal distributions of the predictor
  ## variables other than which.variable.
  xvar <- model$training.data[, which.variable]
  if (is.factor(xvar) || is.character(xvar)) {
    xvar <- as.factor(xvar)
    grid <- as.factor(levels(xvar))
  } else {
    xvar <- as.numeric(xvar)
    grid <- sort(unique(model$training.data[, which.variable]))
    if (length(grid) > gridsize) {
      endpoints <- range(grid)
      grid <- seq(endpoints[1], endpoints[2], length = 50)
    }
  }

  check.nonnegative.scalar(burn)
  if (burn > model$niter) {
    stop(paste0("'burn' cannot exceed the number of MCMC iterations ",
      "used to fit the model."))
  }
  
  draws <- matrix(nrow = model$niter - burn, ncol = length(grid))
  
  for (i in 1:length(grid)) {
    training.data[, which.variable] <- grid[i]
    prediction <- predict(model, newdata = training.data, burn = burn)
    draws[, i] <- rowMeans(prediction)
  }
  
  if (is.null(xlab)) {
    xlab = colnames(training.data)[which.variable]
  }
  if (mean.only) {
    if (is.null(ylab)) {
      ylab <- "Prediction"
    }
    pred <- colMeans(draws)
    if (is.null(ylim)) {
      if (show.points) {
        ylim <- range(pred, model$response)
      } else {
        ylim <- range(pred)
      }
    }
    plot(grid, pred, xlab = xlab, ylab = ylab, ylim = ylim, ...)
  } else {
    ## This branch plots the full predictive distribution.
    if (is.null(ylab)) {
      ylab <- as.character(model$call$formula[[2]])
    }
    if (is.null(ylim)) {
      if (show.points) {
        ylim <- range(draws, model$response)
      } else {
        ylim <- range(draws)
      }
    }
    if (is.factor(grid)) {
      boxplot(draws, xlab = xlab, ylab = ylab, ylim = ylim, pch = 20, ...)
      if (show.points) {
        points(model$training.data[, which.variable],
          model$response,
          ...)
      }


    } else {

      if (show.points) {
        plot(model$training.data[, which.variable], model$response,
          xlab = xlab, ylab = ylab, ylim = ylim, ...)
      }
      PlotDynamicDistribution(draws, grid, xlab = xlab, ylab = ylab, ylim = ylim,
        add = show.points, ...)
    }
  }

  if (report.time) {
    stop.time <- Sys.time();
    cat("Done at   \t",
      format(stop.time),
      "\n")
    cat("Function took",
      difftime(stop.time, start.time, units = "secs"),
      "seconds.\n")
  }
  return(invisible(list(predictive.distribution = draws, x = grid)))
}
