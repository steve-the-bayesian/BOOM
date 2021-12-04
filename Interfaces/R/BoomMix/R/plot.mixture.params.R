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

PlotMixtureParams <- function(model, stem, style = c("density", "ts", "box"),
                              colors = NULL, burn = 0, ...) {
  ## Plots the posterior distribution of parameters in finite mixture
  ## models.
  ##
  ## Args:
  ##   model: An object of class "FiniteMixture", or a list with a
  ##     similar structure.
  ##   stem: A character string giving the name of a parameter or
  ##     mixture component to plot.  The stem can also either include
  ##     the name of a particular parameter in a mixture component, or
  ##     it can be the mixture component name, in which case all the
  ##     parameters associated with that component will be plotted.
  ##   style: A character string indicating the style of plot desired.
  ##     "density" will plot kernel density estimates.  "ts" will plot
  ##     the time series of MCMC draws.  "box" will produce
  ##     side-by-side boxplots.
  ##   colors: An optional character vector specifying the colors to
  ##     use for the different mixture components.
  ##   burn: If burn > 0 then the first 'burn' MCMC draws will be
  ##     discarded.
  ##   ...:  Extra arguments passed to lower level plotting functions
  ##
  ## Returns:
  ##   invisible(NULL)
  ##
  ## Details:
  ##   Some plots for vector- or matrix-valued parameters can only be
  ##   called for one parameter at a time, while multiple
  ##   scalar-valued parameters can be plotted simultaneously.
  style <- match.arg(style)
  nch <- nchar(stem)
  if (substring(stem, nch, nch) != ".") {
    stem <- paste(stem, ".", sep = "")
    nch <- nch + 1
  }
  pos <- grep(stem, names(model))
  leaf <- names(model)[pos]
  num.leaves <- length(leaf)
  has.dot <- length(grep(".", leaf)) > 0
  fields <- strsplit(leaf, ".", fixed = TRUE)

  ## If the variable name has a dot in it, then we need to collapse
  ## the variable names back together, after removing the leaves.
  if (has.dot) {
    variable.names <- sapply(fields,
                             function(x) {paste(head(x, -1), collapse = ".")})
  } else {
    variable.names <- substring(stem, 1, nch-1)
  }
  components <- unique(as.numeric(sapply(fields, function(x) {tail(x, 1)})))
  number.of.components <- length(components);

  if (is.null(colors)) {
    ## skip black
    colors <- 1:number.of.components + 1
  }

  unique.variable.names <- unique(variable.names)
  nvars <- length(unique.variable.names)
  number.of.rows <- floor(sqrt(nvars))
  number.of.cols <- ceiling(nvars / number.of.rows)
  opar <- par(mfrow = c(number.of.rows, number.of.cols))
  on.exit(par(opar))

  if (length(colors) > number.of.components) {
    colors <- colors[1:number.of.components]
  }
  for (i in 1:nvars) {
    current.pos <- pos[variable.names == unique.variable.names[i]]
    if (is.matrix(model[[current.pos[1]]])) {
      ##------------------------------------------------------------
      ## This block handles vector parameters.
      if (style == "box") {
        CompareVectorBoxplots(model[current.pos],
                              main = unique.variable.names[i],
                              colors = colors,
                              ...)
      }
      if (nvars > 1) {
        stop("You can only call PlotMixtureParams with multiple variables,",
             "where one or more is vector-valued, unless style == 'box'.")
      }
      else if (style == "ts") {
        CompareManyTs(model[current.pos],
                      main = unique.variable.names[i],
                      color = colors,
                      burn = burn,
                      ...)
      } else if (style == "density") {
        CompareManyDensities(model[current.pos],
                             main = unique.variable.names[i],
                             color = colors,
                             burn = burn,
                             ...)
      }
    } else if (is.array(model[[current.pos[1]]])) {
      ##------------------------------------------------------------
      ## This block handles matrix parameters.
      if (nvars > 1) {
        stop("You can't call PlotMixtureParams with multiple variables",
             "if one is matrix valued.")
      }
      if (style == "ts") {
        CompareManyTs(model[current.pos],
                      main = unique.variable.names[i],
                      color = colors,
                      burn = burn,
                      ...)
      } else if (style == "box" || style == "density") {
        CompareManyDensities(model[current.pos],
                             main = unique.variable.names[i],
                             color = colors,
                             burn = burn,
                             ...)
      }
    } else {
      ##------------------------------------------------------------
      ## Plots for scalar quantities go here
      draws <- as.matrix(as.data.frame(model[current.pos]))
      if (burn > 0) draws <- draws[-(1:burn), ]

      if (style == "density") {
        CompareDensities(draws,
                         main = unique.variable.names[i],
                         legend.text = paste(sort(components)),
                         legend.title = "Component",
                         col = colors,
                         ...)
      } else if (style == "ts") {
        plot.ts(draws,
                plot.type = "single",
                col = colors,
                lty = 1:number.of.components,
                main = unique.variable.names[i],
                ...)
      } else if (style == "box") {
        boxplot(as.data.frame(draws),
                col = colors,
                main = unique.variable.names[i],
                ...)
      }
    }
  }
  return(invisible(NULL))
}
