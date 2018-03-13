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

summary.bsts <- function(object, burn = SuggestBurn(.1, object), ...) {
  ## Prints a summary of the supplied bsts object.
  ## Args:
  ##   object:  An object of class 'bsts'
  ##   burn:  The number of MCMC iterations to discard as burn-in.
  ##   ...: Additional arguments passed to summary.lm.spike, if
  ##     'object' has a regression component.
  ## Returns:
  ##   A list of summaries describing the bsts object.
  ##   residual.sd: The posterior mean of the residual standard
  ##     deviation parameter.
  ##   prediction.sd: The standard deviation of the one-step-ahead
  ##     prediction errors.  These differ from the residuals because
  ##     they only condition on the data preceding the prediction.
  ##     The residuals condition on all data in both directions.
  ##   rquare: The R-square from the model, computing by comparing
  ##     'residual.sd' to the sample variance of the original series.
  ##   relative.gof: Harvey's goodness of fit statistic:
  ##     1 - SSE(prediction errors) / SST(first difference of original series).
  ##     This is loosly analogous to the R^2 in a regression model.
  ##     It differs in that the baseline model is a random walk with
  ##     drift (instead of the sample mean).  Models that fit worse,
  ##     on average, than the baseline model can have a negative
  ##     relative.gof score.
  ##   size: If the original object had a regression component, then
  ##     'size' summarizes the distribution of the number of nonzero
  ##     coefficients.
  ##   coefficients: If the original object had a regression
  ##     component, then 'coef' contains a summary of the regression
  ##     coefficients computed using summary.lm.spike.
  stopifnot(inherits(object, "bsts"))
  sigma.obs <- object$sigma.obs
  if (!is.null(sigma.obs)) {
    if (burn > 0) {
      sigma.obs <- sigma.obs[-(1:burn)]
    }
    residual.sd <- mean(sigma.obs)
    original.variance <- var(object$original.series, na.rm = TRUE)
    stopifnot(original.variance > 0)
    rsquare <- 1 - residual.sd^2 / original.variance
  }

  prediction.errors <- bsts.prediction.errors(object, burn = burn)$in.sample
  prediction.sse <- sum(colMeans(prediction.errors)^2)
  original.series <- as.numeric(object$original.series)
  dy <- diff(original.series)
  prediction.sst <- var(dy) * (length(dy) - 1)

  ans <- list(residual.sd = residual.sd,
              prediction.sd = sd(colMeans(prediction.errors)),
              rsquare = rsquare,
              relative.gof = 1 - prediction.sse / prediction.sst)

  ##----------------------------------------------------------------------
  ## summarize the regression coefficients
  if (object$has.regression) {
    beta <- object$coefficients
    if (burn > 0) {
      beta <- beta[-(1:burn), , drop = FALSE]
    }
    include <- beta != 0
    model.size <- rowSums(include)
    ans$size <- summary(model.size)
    ans$coefficients <- SummarizeSpikeSlabCoefficients(object$coefficients,
                                                       burn = burn, ...)
  }
  return(ans)
}
