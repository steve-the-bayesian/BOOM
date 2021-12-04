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

PlotDensityDistribution <- function(model, burn = NULL, xlim = NULL, xlab=NULL, ...) {
  ## If the only mixture component in the model is a NormalMixtureComponent,
  ## then
  ##
  ## Args:
  ##   model: An object of class "FiniteMixture", or a list with a
  ##     similar structure.
  ##   burn: If burn > 0 then the first 'burn' MCMC draws will be
  ##     discarded.
  ##   ...:  Extra arguments passed to PlotDynamicDistribution.
  ##
  ## Returns:
  ##   invisible(NULL)
  ##
  spec <- model$mixture.component.specification
  okay <- (length(spec) == 1 && inherits(spec[[1]], "NormalMixtureComponent"))
  if (!okay) {
    stop("Density plots are only available for NormalMixtureComponent models.")
  }

  weights <- model$mixing.weights
  ncomp <- ncol(weights)
  niter <- nrow(weights)

  mu <- matrix(nrow = niter, ncol = ncomp)
  sigma <- matrix(nrow = niter, ncol = ncomp)
  for (i in 1:ncomp) {
    mu.name <- paste0("mu.", i-1)
    mu[, i] <- model[[mu.name]]

    sigma.name <- paste0("sigma.", i-1)
    sigma[, i] <- model[[sigma.name]]
  }

  if (is.null(xlab)) {
    xlab <- spec$name
  }
  if (is.null(xlab)) {
    xlab <- ""
  }

  if (is.null(burn)) {
    burn <- SuggestBurnLogLikelihood(model$log.likelihood)
  }

  if (burn > 0) {
    mu <- mu[-(1:burn), ]
    sigma <- sigma[-(1:burn), ]
    weights <- weights[-(1:burn), ]
    niter <- niter - burn
  }

  if (is.null(xlim)) {
    lower.limit <- min(mu - 3 * sigma)
    upper.limit <- max(mu + 3 * sigma)
    xlim <- c(lower.limit, upper.limit)
  }

  stopifnot(is.numeric(xlim), length(xlim) == 2)
  xlim <- sort(xlim)

  x.arg <- seq(xlim[1], xlim[2], len=100)
  density.values <- array(dim = c(niter, length(x.arg), ncomp))
  nx <- length(x.arg)

  for (s in 1:ncomp) {
    density.values[, , s] <- matrix(dnorm(
      rep(x.arg, niter),
      rep(mu[, s], each = nx),
      rep(sigma[, s], each = nx)) * rep(weights[, s], each = nx),
      nrow = niter, byrow=TRUE)
  }

  density.distribution <- apply(density.values, c(1, 2), sum)

  PlotDynamicDistribution(density.distribution, timestamps = x.arg,
    xlab = xlab, ...)
  abline(h=0, lty=3, col="lightgray")

  ans <- list(density = density.distribution, x = x.arg)

  return(invisible(ans))
}
