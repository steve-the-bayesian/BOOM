# Copyright 2021 Steven L. Scott. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA


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
