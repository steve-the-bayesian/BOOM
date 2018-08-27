library(bsts)
library(testthat)

test_that(paste0("Prediction with olddata, but without regression, ",
  "includes the right original series."), {
    data(AirPassengers)
    y <- log(AirPassengers)
    ss <- AddLocalLinearTrend(list(), y)
    ss <- AddSeasonal(ss, y, nseasons = 12)
    model <- bsts(y, state.specification = ss, niter = 500, ping = -1)
    full.pred <- predict(model, horizon = 12, burn = 100)

    training <- window(y, end = c(1959, 12))
    small.model <- bsts(training, state.specification = ss, niter = 500, ping = -1)
    pred.holdout <- predict(small.model, horizon = 12)
    expect_equal(length(pred.holdout$original.series), length(y) - 12)
    
    updated.pred <- predict(small.model, horizon = 12, olddata = y)
    expect_equal(updated.pred$original.series, y)
    expect_equal(length(updated.pred$original.series), length(full.pred$original.series))
  })
