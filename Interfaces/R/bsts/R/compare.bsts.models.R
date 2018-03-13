# Copyright 2018 Google LLC. All Rights Reserved.
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

CompareBstsModels <- function(model.list,
                              burn = SuggestBurn(.1, model.list[[1]]),
                              filename = "",
                              colors = NULL,
                              lwd = 2,
                              xlab = "Time",
                              main = "",
                              grid = TRUE,
                              cutpoint = NULL) {
  ## Produce a set of line plots showing the cumulative absolute one
  ## step ahead prediction errors for different models.  This plot not
  ## only shows which model is doing the best job predicting the data,
  ## it highlights regions of the data where the predictions are
  ## particularly good or bad.
  ##
  ## Args:
  ##   model.list:  A list of bsts models.
  ##   burn: The number of initial MCMC iterations to remove from each
  ##     model as burn-in.
  ##   filename: A string.  If non-empty string then a pdf of the
  ##     plot will be saved in the specified file.
  ##   colors: A vector of colors to use for the different lines in
  ##     the plot.  If NULL then the rainbow pallette will be used.
  ##   lwd: The width of the lines to be drawn.
  ##   xlab: Labels for the horizontal axis.
  ##   main: Main title for the plot.
  ##   grid: Logical.  Should gridlines be drawn in the background?
  ##   cutpoint: Either NULL, or an integer giving the observation number used
  ##     to define a holdout sample.  Prediction errors occurring after the
  ##     cutpoint will be true out of sample errors.  If NULL then all
  ##     prediction errors are "in sample".  See the discussion in
  ##     'bsts.prediction.errors'.
  ##
  ## Returns:
  ##   Invisibly returns the matrix of cumulative errors (the lines in
  ##   the top panel of the plot).
  stopifnot(is.list(model.list))
  stopifnot(length(model.list) > 1)
  stopifnot(all(sapply(model.list, inherits, "bsts")))
  if (HasDuplicateTimestamps(model.list[[1]])) {
    stop("CompareBstsModels does not support duplicate timestamps.")
  }
  time.dimension <-
    sapply(model.list, function(m) {dim(m$state.contributions)[3]})
  stopifnot(all(time.dimension == time.dimension[1]))

  model.names <- names(model.list)
  if (is.null(model.names)) {
    model.names <- paste("Model", 1:length(model.list))
  }
  number.of.models <- length(model.list)
  if (filename != "") pdf(filename)
  opar <- par(mfrow=c(2, 1))
  original.margins <- c(5.1, 4.1, 4.1, 2.1)
  margins <- original.margins
  opar$mar <- original.margins
  margins[1] <- 0
  par(mar = margins)
  errors <- bsts.prediction.errors(model.list[[1]], burn = burn)$in.sample
  cumulative.errors <- matrix(nrow = number.of.models, ncol = ncol(errors))
  for (i in 1:number.of.models) {
    if (is.null(cutpoint)) {
      prediction.errors <- bsts.prediction.errors(
          model.list[[i]], burn = burn)$in.sample
    } else {
      prediction.errors <- bsts.prediction.errors(
          model.list[[i]], burn = burn, cutpoints = cutpoint)[[1]]
    }
    cumulative.errors[i, ] <- cumsum(abs(colMeans(prediction.errors)))
  }
  if (is.null(colors)) colors <- c("black", rainbow(number.of.models-1))

  idx <- model.list[[1]]$timestamp.info$timestamps
  plot(zoo(cumulative.errors[1, ], order.by = idx),
       ylim = range(cumulative.errors),
       ylab = "cumulative absolute error",
       xaxt = "n",
       lwd = lwd,
       col = colors[1],
       yaxs = "i",
       main = main)
  axis(2)
  for (i in 2:number.of.models) {
    lines(zoo(cumulative.errors[i, ], order.by = idx),
          lty = i,
          col = colors[i],
          lwd = lwd)
  }

  if (grid) {
    grid()
  }

  legend("topleft",
         model.names,
         lty = 1:number.of.models,
         col = colors,
         bg  = "white",
         lwd = lwd)

  margins <- original.margins
  margins[3] <- 0
  par(mar = margins)
  plot(model.list[[1]]$original.series,
       main = "",
       ylab = "scaled values",
       xlab = xlab,
       yaxs = "i")
  if (grid) {
    grid()
  }
  par(opar)
  if (filename != "") dev.off()
  return(invisible(cumulative.errors))
}
