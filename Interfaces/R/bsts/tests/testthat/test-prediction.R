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
