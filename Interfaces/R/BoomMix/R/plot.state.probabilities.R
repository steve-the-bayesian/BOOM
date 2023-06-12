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
