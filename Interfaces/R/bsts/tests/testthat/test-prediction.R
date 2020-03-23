library(bsts)
library(testthat)

test_that("Prediction with olddata, but without regression.", {
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 200, ping = -1)
  full.pred <- predict(model, horizon = 12, burn = 100)

  training <- window(y, end = c(1959, 12))
  small.model <- bsts(training, state.specification = ss, niter = 200, ping = -1)
  pred.holdout <- predict(small.model, horizon = 12)
  expect_equal(length(pred.holdout$original.series), length(y) - 12)

  updated.pred <- predict(small.model, horizon = 12, olddata = y)
  expect_equal(updated.pred$original.series, y)
  expect_equal(length(updated.pred$original.series), length(full.pred$original.series))
})


test_that("Prediction with olddata, and with regression. ", {
  data(iclaims)
  training <- initial.claims[1:402, ]
  holdout1 <- initial.claims[403:450, ]
  holdout2 <- initial.claims[451:456, ]

  ss <- AddLocalLinearTrend(list(), training$iclaimsNSA)
  ss <- AddSeasonal(ss, training$iclaimsNSA, nseasons = 52)
  model <- bsts(iclaimsNSA ~ ., state.specification = ss, data =
                training, niter = 100)

  ## Predict the holdout set given the training set.
  ## This is really fast, because we can use saved state from the MCMC
  ## algorithm.
  pred.full <- predict(model, newdata = rbind(holdout1, holdout2))
  expect_equal(length(pred.full), 5)
  expect_equal(names(pred.full), c("mean", "median", "interval",
    "distribution", "original.series"))
  expect_equal(length(pred.full$mean), 54)
  expect_true(is.matrix(pred.full$distribution))
  expect_equal(ncol(pred.full$distribution), 54)

  ## Predict holdout 2, given training and holdout1.
  ## This is much slower because we need to re-filter the 'olddata' before
  ## simulating the predictions.
  pred.update <- predict(model, newdata = holdout2,
    olddata = rbind(training, holdout1))

  expect_equal(length(pred.update), 5)
  expect_equal(names(pred.update), c("mean", "median", "interval",
    "distribution", "original.series"))
  expect_equal(length(pred.update$mean), 6)
  expect_true(is.matrix(pred.update$distribution))
  expect_equal(ncol(pred.update$distribution), 6)
})

test_that("Off by one error is solved", {
  # Predictions for regression models experienced an off-by-one error.  This
  # test demonstrates that the error is solved.

  ## Simulate data.  The pattern is a just a day-of-week pattern with
  ## substantial day-to-day variation.
  N = 7*52
  dateseq <- seq(as.Date("2000-01-02"), length.out = N, by = "days")
  proportions <- c(0.005, 0.14, 0.17, 0.23, 0.265, 0.18, 0.01)
  pMatrix <- matrix(0, ncol = 1, nrow = N)
  for (i in 1:7) {
    day.indicator <- which(weekdays(dateseq, FALSE) == weekday.names[i])
    pMatrix[day.indicator, 1] = proportions[i]
  }
  set.seed(102)
  weekly.sim <- arima.sim(list(ma=-0.1), n = 52, sd = 0.01)
  weekly.sim <- exp(diffinv(weekly.sim, differences = 1, xi = log(200)))[-1]
#  plot(weekly.sim, type = "l")

  seqindx <- c(seq(1, N, by =  7), N + 1)
  daily.sim.allocate <- do.call(rbind, lapply(1:(length(seqindx) - 1), function(i){
    cbind(pMatrix[seqindx[i]:(seqindx[i + 1] - 1)] * weekly.sim[i])
  }))
  # add a holiday effect
  independence <- which(dateseq == "2000-07-04")
  daily.sim.allocate[independence] <- daily.sim.allocate[independence] * 0.1
  xreg <- rep(0, N)
  xreg[independence] <- 1

  # colnames(xreg) <- "independence"
  dat <- data.frame(y = as.numeric(daily.sim.allocate), independence = xreg,
    days = weekdays(dateseq), stringsAsFactors = FALSE)

  rownames(dat) <- dateseq
  # dat <- zoo(dat, dateseq)
  # plot(dat[,1], type = "l")
  tail(dat[1:(independence+24),])

  ## estimation part
  spec <- AddLocalLevel(list(), y = as.numeric(dat[1:(independence+24),1]))
  spec <- AddSeasonal(spec, y = as.numeric(dat[1:(independence+24),1]),
    nseasons = 7)

  train <- 1:(independence + 24)

  # estimate with regressor and intercept
  niter <- 250
  seed <- 8675309
  mod1 <- bsts(y~independence, data = dat[train, 1:2],
    state.specification = spec, niter = niter, seed = seed)

  # estimate with regressor and no intercept
  ## mod2 <- bsts(y~independence-1, data = dat[train, 1:2],
  ##   state.specification = spec, niter = niter, seed = seed)

  # estimate with no regressor and no intercept
  mod3 <- bsts(as.numeric(dat[train, 1]),
    state.specification = spec, niter = niter, seed = seed)

  ## forecast part
  test <- (independence + 25):N
  horizon <- length(test)
  # the newdata has only zeros.
  newdata <- dat[test, 2, drop = FALSE]

  p1 <- predict(mod1, newdata = newdata, seed = seed)
#  p2 <- predict(mod2, newdata = newdata, seed = seed)
  p3 <- predict(mod3, horizon = horizon, seed = seed)

  # Comparison of output.  The predictions from models 1 and 2 (which have a
  # regression component) are off by one when compared to the model sans
  # regression component.
  comparison <- data.frame(
    day = weekdays(as.Date(rownames(dat[test, 2, drop = FALSE]))),
    reg1 = colMeans(p1$distribution),
#    reg2 = colMeans(p2$distribution),
    ts = colMeans(p3$distribution),
    actual = dat[test, 1])

  ## > head(comparison)
  ##         day      reg1      reg2       ts     actual
  ## 1  Saturday  3.034669  3.001731  2.038222  2.153567
  ## 2    Sunday  1.971062  1.922798  1.437442  1.074628
  ## 3    Monday 30.622732 30.597797 29.827300 30.089592
  ## 4   Tuesday 36.785562 36.711367 34.199968 36.537362
  ## 5 Wednesday 49.437153 49.367559 48.581829 49.432902
  ## 6  Thursday 56.878702 56.844302 55.889856 56.955300

  # In the error state the first row showed reg1 and reg2 with very large values
  # because they were using an off-by-1 seasonal effect
  expect_true(all(abs(comparison$reg1 - comparison$ts) < 4))
  expect_true(cor(comparison$reg1, comparison$ts) > .98)
})
