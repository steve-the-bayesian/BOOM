BartPartialDependencePlot <- function(model,
                                      which.variable,
                                      burn = SuggestBartBurn(model),
                                      data.fraction = .2,
                                      thin = 10,
                                      mean.only = FALSE,
                                      show.points = FALSE,
                                      xlab = NULL,
                                      ylab = NULL,
                                      ylim = NULL,
                                      ...) {
  ## Args:
  ##   model:  The model object returned by BoomBart.
  ##   which.variable: Either the name of the variable for which a
  ##     partial dependence plot is desired, or its index (column
  ##     number) in the design matrix.
  ##   burn: The number of initial MCMC iterations to discard as
  ##     burn-in.
  ##   data.fraction: The fraction of observations in the design
  ##     matrix to use when constructing the partial dependence plot.
  ##     A random sub-sample of this fraction will be taken (without
  ##     replacement).
  ##   thin: The frequency of MCMC iterations to keep after burn-in.
  ##     For example, setting thin = 10 would keep every 10th
  ##     iteration.
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
  design.matrix <- model$design.matrix
  if (data.fraction < 1) {
    rows <- sample(1:nrow(design.matrix),
                   size = round(data.fraction * nrow(design.matrix)))
    design.matrix <- design.matrix[rows, , drop = FALSE]
  }

  start.time <- Sys.time()
  cat("Starting at\t", format(start.time), "\n")

  if (is.character(which.variable)) {
    original.variable.name <- which.variable
    which.variable <- pmatch(which.variable,
                             colnames(model$design.matrix))
    if (is.na(which.variable)) {
      stop(original.variable.name, "was not uniquely matched in",
           colnames(model$design.matrix))
    }
  }

  posterior.predictive.distribution <-
    .Call("boom_bart_partial_dependence_plot_wrapper_",
          model,
          as.integer(which.variable),
          design.matrix,
          burn,
          thin,
          PACKAGE = "BoomBart")

  x <- posterior.predictive.distribution$x

  family <- model$family
  if (family != "gaussian" && scale == "data") {
    inverse.link <- c("poisson" = exp,
                      "probit" = pnorm,
                      "logit" = plogis)[family]
    posterior.predictive.distribution <-
      inverse.link(posterior.predictive.distribution$draws);
  }

  if (is.null(xlab)) {
    xlab = colnames(design.matrix)[which.variable]
  }
  if (mean.only) {
    if (is.null(ylab)) {
      ylab <- "Prediction"
    }
    pred <- colMeans(posterior.predictive.distribution$draws)
    if (is.null(ylim)) {
      ylim <- range(pred)
    }
    plot(x, pred, xlab = xlab, ylab = ylab, ylim = ylim, ...)
  } else {
    if (is.null(ylab)) {
      ylab = "Distribution"
    }
    if (is.null(ylim)) {
      ylim <- range(posterior.predictive.distribution$draws)
    }
    PlotDynamicDistribution(posterior.predictive.distribution$draws,
                            x,
                            xlab = xlab,
                            ylab = ylab,
                            ylim = ylim,
                            ...)
  }

  if (show.points) {
    points(model$design.matrix[, which.variable],
           model$response,
           ...)
  }

  stop.time <- Sys.time();
  cat("Done at   \t",
      format(stop.time),
      "\n")
  cat("Function took",
      difftime(stop.time, start.time, units = "secs"),
      "seconds.\n")
  return(invisible(list(posterior.predictive.distribution =
                        posterior.predictive.distribution,
                        x = x)))
}
