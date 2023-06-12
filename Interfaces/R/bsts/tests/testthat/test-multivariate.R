library(bsts)
library(testthat)

# GDP figures for 57 countries as reported by the OECD.
## data(gdp)
## series.id <- gdp$Country
## timestamps <- gdp$Time

test_that("Predictions and state are sane when only factors are present", {
  nobs <- 200
  ndim <- 3
  nfactors <- 2
  seed <- 8675309
  set.seed(seed)

  residual.sd <- sqrt(abs(rnorm(ndim))) / 10

  factors <- matrix(rnorm(nobs * nfactors, sd=1), ncol = nfactors)
  factors <- apply(factors, 2, cumsum)
  coefficients <- matrix(rnorm(ndim * nfactors), nrow=nfactors)
  coefficients <- coefficients/coefficients[, 1]

  state <- factors %*% coefficients
  errors <- matrix(rnorm(nobs * ndim), ncol=ndim) %*% diag(residual.sd)
  y <- state + errors

  ss <- AddSharedLocalLevel(list(), y, nfactors=nfactors)
  x <- matrix(rep(1, nobs), ncol=1)

  model <- mbsts(y, ss, niter=500, data.format="wide", seed=seed)
  pred <- predict(model, 24, seed = seed)
  ## Each time series should be within the prediction interval of the next
  ## point.
  for (s in 1:ndim) {
    last.y = tail(y[, s], 1)
    interval <- pred$interval[s, , 1]
    expect_gt(last.y, interval[1])
    expect_lt(last.y, interval[2])
  }

  state.means <- apply(model$shared.state.contributions, c(1, 3, 4), sum)
  intervals <- apply(state.means[-(1:100), , ], c(2,3), quantile, c(.025, .975))
  state.posterior.means <- apply(state.means[-(1:100), , ], c(2,3), mean)

  mean.residual.sd <- colMeans(model$residual.sd[-(1:100), ])
  for (s in 1:ndim) {
    expect_gt(cor(y[, s], colMeans(state.means[, s, ])), .99)
  }
})
