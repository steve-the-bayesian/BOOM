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


PlotMixingWeights <- function(model, style = c("boxplot", "ts", "density"),
                              burn = 0, ylim = c(0, 1), color = NULL, ...) {
  ## PLot the MCMC draws of the mixing weights.
  ## Args:
  ##   model:  An object of class FiniteMixture.
  ##   style:  A string describing the desired type of plot.
  ##   burn: The number of MCMC iterations to be discarded before
  ##     making the plot.
  ##   ylim:  The limits on the y axis.
  ##   color:  A vector of colors to use for the different mixture components.
  ##   ...:  Extra arguments passed to lower level plotting functions.
  ## Returns:
  ##   invisible(NULL)
  weights <- model$mixing.weights
  if (burn > 0) weights <- weights[-(1:burn), ]
  style <- match.arg(style)
  number.of.components <- ncol(weights)
  vnames <- paste(1:number.of.components - 1)
  if (is.null(color)) {
    color <- (1:ncol(weights)) + 1  # Skip black.
  }

  if (style == "boxplot") {
    boxplot(weights, xlab = "Component", ylab = "Mixing Weights",
            ylim = ylim, names = vnames, col = color, ...)
    abline(h = c(0, 1), lty = 3)
  } else if (style == "ts") {
    plot.ts(weights, plot.type = "single", lty = 1:ncol(weights),
            ylim = ylim, col = color,
            xlab = "Iteration", ylab = "Mixing Weights", ...)
    abline(h = c(0, 1), lty = 3)
    legend("topright", col= color, lty = 1:ncol(weights),
           title = "Component", legend = vnames, bg = "white")
  } else if (style == "density") {
    CompareDensities(weights, legend.title = "Component",
                     xlab = "Mixing Weights", ylab = "Density",
                     col = color, ...)
  }
  return(invisible(NULL))
}
