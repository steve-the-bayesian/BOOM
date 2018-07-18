library(BoomSpikeSlab)
library(testthat)

nobs <- 1000
xdim <- 20
predictors <- matrix(rnorm(nobs * xdim), nrow = nobs, ncol = xdim)
## There are three sets of predictors which are perfectly correlated with one
## another.
predictors[, 2] <- 2.3 * predictors[, 1]
predictors[, 3] <- 1.7 * predictors[, 2]

## The first coefficient has a 
beta <- c(14.0, 0, 0, 21.7, -3.8, 14.9, rep(0, xdim - 6))
residual.sd <- .4
response <- 1.9 + rnorm(nobs, predictors %*% beta, residual.sd)

## Model 1 has a swap move.  Predictors with absolute correlation .8 or higher
## are considered.
model1 <- lm.spike(response ~ predictors, niter = 1000)

## Model 2 turns off the swap move by setting an impossible threshold.
model2 <- lm.spike(response ~ predictors, niter = 1000, model.options = SsvsOptions(
  correlation.swap.threshold = 2.0))

## From visual inspection it appears the swap move doesn't help all that much,
## but that's only because the ordinary SSVS works so well.
