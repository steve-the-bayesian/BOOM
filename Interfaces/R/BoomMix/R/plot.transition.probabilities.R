PlotTransitionProbabilities <- function(hmm,
                                        style = c("ts", "box", "acf"),
                                        burn = 0,
                                        ylim = c(0, 1),
                                        colors = NULL,
                                        ...) {
  ## Plots the posterior distribution of the transition probability
  ## matrix for the hidden Markov chain.
  ## Args:
  ##   hmm:  An object of class HiddenMarkovModel.
  ##   style:  The style of plot to produce.
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   ylim:  Limits on the vertical axis.
  ##   colors:  A vector of colors to use for the plots.
  ##   ...: Extra arguments passed to PlotManyTs, PlotMacf, or BoxplotTrue.
  ## Returns:
  ##   Called for its side effect.
  style <- match.arg(style)
  probs <- hmm$transition.probabilities
  if (burn > 0) {
    probs <- probs[-(1:burn), , , drop = FALSE]
  }
  stopifnot(is.array(probs) && length(dim(probs) == 3))
  number.of.states <- dim(probs)[2]
  if (is.null(colors)) {
    colors <- 1:number.of.states + 1  ## match plot.mixture.params.R
  }
  stopifnot(length(colors) == number.of.states)

  if (style == "ts") {
    PlotManyTs(probs, ylim = ylim, color = rep(colors, number.of.states), ...)
  } else if (style == "acf") {
    PlotMacf(probs, ...)
  } else if (style == "box") {
    BoxplotMcmcMatrix(probs, ylim = ylim, colors = colors, ...)
  }
}
