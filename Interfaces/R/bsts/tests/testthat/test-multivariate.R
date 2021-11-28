library(bsts)
library(testthat)

# GDP figures for 57 countries as reported by the OECD.
data(gdp)
series.id <- gdp$Country
timestamps <- gdp$Time

test_that("Wide long conversion", {
  wide <- matrix(rnorm(40), nrow = 10, ncol = 4)
  long <- WideToLong(wide)
  wide2 <- LongToWide(long$values, long$series, long$time)
  expect_equal(sum((wide-wide2)^2), 0.0)

  wide.with.na <- wide
  wide.with.na[c(1, 3, 7), c(2, 3)] <- NA
  long.with.na <- WideToLong(wide.with.na)
  wide.with.na2 <- LongToWide(long.with.na$values, long.with.na$series,
    long.with.na$time)
})


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
  model <- mbsts(y, ss, niter=250, data.format="wide", seed=seed)
  pred <- predict(model, 24, seed = seed)
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
    expect_gt(corr(y[, s], state.means[s, ]), .99)
  }

})
