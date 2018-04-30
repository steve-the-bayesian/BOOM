PlotInitialStateDistribution <- function(hmm,
                                         style = c("ts", "box", "acf"),
                                         burn = 0,
                                         ylim = c(0, 1),
                                         colors = NULL,
                                         ...) {
  ## Plots the initial state distribution of the hidden Markov chain.
  ## Args:
  ##   hmm:  An object of class HiddenMarkovModel.
  ##   style:  The style of plot to produce.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   ylim:  Limits on the vertical axis.
  ##   ...: Extra arguments passed to PlotManyTs, PlotMacf, or BoxplotTrue.
  ## Returns:
  ##   Called for its side effect.
  style <- match.arg(style)
  probs <- hmm$initial.state.distribution
  if (burn > 0) {
    probs <- probs[-(1:burn), , drop = FALSE]
  }
  if (is.null(colors)) {
    colors <- 1:ncol(probs) + 1 ## match color scheme from plot.mixture.params.R
  }
  stopifnot(is.array(probs) && length(dim(probs) == 2))
  if (style == "ts") {
    PlotManyTs(probs, ylim = ylim, color = colors, ...)
  } else if (style == "acf") {
    PlotMacf(probs, ...)
  } else if (style == "box") {
    BoxplotTrue(probs, ylim = ylim, color = colors, ...)
  }
}
