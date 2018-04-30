# Copyright 2012 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

PlotStateProbabilities <- function(model,
                                   group.id = NULL,
                                   colors = NULL,
                                   xlab = "Observation",
                                   ylab = "Probability",
                                   components = NULL,
                                   ...) {
  ## Plots the class membership probabilities for each observation
  ## used to fit 'model.'
  ## Args:
  ##   model:  An object inheriting from 'FiniteMixture'.
  ##   group.id: The specific group for which state probabilities
  ##     should be plotted.  If NULL then all groups will be plotted.
  ##     If you did not specify 'group.id' when fitting 'model' then
  ##     this parameter can be ignored.
  ##   colors: A vector of colors to use for the component plots.
  ##     This can help make associations between these plots and
  ##     (e.g.) those produced by PlotMixtureParams.
  ##   xlab:  The label for the horizontal axis.
  ##   ylab:  The label for the vertical axis.
  ##   components: A numeric vector listing which components should be
  ##     plotted.  Components are numbered starting from 0.
  ##   ...:  Extra arguments passed to 'plot'.
  ##
  ## Returns:
  ##   Called for its side effect.
  if (is.null(group.id)) probs <- model$state.probabilities
  else probs <- model$state.probabilities[[group.id]]
  stopifnot(is.matrix(probs))

  number.of.available.components <- ncol(probs)

  if (is.null(colors)) {
    colors <- 1:number.of.available.components + 1
  }

  if (is.null(components)) {
    components <- (1:number.of.available.components) - 1
  }

  number.of.requested.components <- length(components)
  nr <- floor(sqrt(number.of.requested.components))
  nc <- ceiling(number.of.requested.components / nr)
  opar <- par(mfrow = c(nr, nc))
  on.exit(par(opar))

  ## It is annoying to shift 'components' back and forth between
  ## 0-based and 1-based indexing.  TODO(stevescott): Fix the c-code
  ## to index components from 1.
  for (component in (components + 1)) {
    ## The index 'component' counts components starting from one.
    plot(probs[, component],
         xlab = xlab, ylab = ylab,
         main = paste("Component", component - 1),
         col = colors[component],
         ...)
  }
  return(invisible(NULL))
}
