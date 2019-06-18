library(testthat)
library(bsts)
seed <- 8675309
set.seed(seed)

trend <- cumsum(rnorm(1095, 0, .1))
dates <- seq.Date(from = as.Date("2014-01-01"), length = length(trend), by = "day")
y <- zoo(trend + rnorm(length(trend), 0, .2), dates)

AddHolidayEffect <- function(y, dates, effect) {
  ## Adds a holiday effect to simulated data.
  ## Args:
  ##   y: A zoo time series, with Dates for indices.
  ##   dates: The dates of the holidays.
  ##   effect: A vector of holiday effects of odd length.  The central effect is
  ##     the main holiday, with a symmetric influence window on either side.
  ## Returns:
  ##   y, with the holiday effects added.
  time <- dates - (length(effect) - 1) / 2
  for (i in 1:length(effect)) {
    y[time] <- y[time] + effect[i]
    time <- time + 1
  }
  return(y)
}

## Define some holidays.
memorial.day <- NamedHoliday("MemorialDay")
memorial.day.effect <- c(-.75, -2, -2)
memorial.day.dates <- as.Date(c("2014-05-26", "2015-05-25", "2016-05-30"))
y <- AddHolidayEffect(y, memorial.day.dates, memorial.day.effect)

## The holidays can be in any order.
holiday.list <- list(memorial.day)

## Let's train the model to just before MemorialDay
cut.date = as.Date("2016-05-25")
train.data <- y[time(y) < cut.date]
test.data <- y[time(y) >= cut.date]
ss <- AddLocalLevel(list(), train.data)
ss <- AddRegressionHoliday(ss, train.data, holiday.list = holiday.list)
model <- bsts(train.data, state.specification = ss, niter = 100, ping = 0,
  seed = seed)

## Now make a prediction covering MemorialDay
cat("Starting the prediction.\n")
my.horizon <- 15
pred <- predict(object = model, horizon = my.horizon, seed = seed)
plot(pred, plot.original = 365)
points(index(test.data), test.data)

test_that("Holiday covers true values", {
  expect_true(CheckMcmcMatrix(pred$distribution, test.data[1:15]))
})
