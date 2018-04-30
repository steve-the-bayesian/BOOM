NumberOfTrees <- function(model) {
  ## Args:
  ##   model:  An object of class BoomBart.
  ##
  ## Returns:
  ##   A numeric vector containing the number of tree in each MCMC
  ##   iteration
  model$tree.size.distribution[, 1]
}

PlotTreeSizeDistribution <- function(model, burn = 0) {
  ## Args:
  ##   model:  An object of class BoomBart
  ##   burn:  A number of MCMC iterations to discard as burn-in.
  if (burn > 0) {
    object <- object[-(1:burn), ]
  }
  opar <- par(mfrow = c(1, 2))
  on.exit(par(opar))
  plot.ts(model$tree.size.distribution[, 1],
          xlab = "MCMC Iteration",
          ylab = "Number of Trees")

  colors <- c("black", "red", "blue", "green", "blue", "red", "black")

  plot.ts(model$tree.size.distribution[, -1],
          plot.type = "single",
          col = colors,
          xlab = "MCMC Iteration",
          ylab = "Nodes / Tree")
  legend("topleft",
         legend = rev(c("min", "10%", "25%", "50%", "75%", "90%", "max")),
         col= colors,
         lty = rep(1, 7))
  return(invisible(NULL))
}
