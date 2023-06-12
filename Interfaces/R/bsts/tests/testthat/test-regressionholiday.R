library(bsts)
library(testthat)

context("test-regressionholiday.R")

set.seed(8675309)

trend <- cumsum(rnorm(730, 0, .1))
dates <- seq.Date(from = as.Date("2014-01-01"), length = length(trend),
  by = "day")
y <- zoo(trend + rnorm(length(trend), 0, .2), dates)

McmcMatrixReport <- function(draws, truth, confidence = .95) {
  ## TODO(stevescott): Remove this after Boom 0.8.1 or later is published.  Boom
  ## now includes this function to be used with CheckMcmcMatrix.
  alpha <- 1 - confidence
  intervals <- t(apply(draws, 2, quantile, c(alpha / 2, (1 - alpha / 2))))
  ans <- cbind(intervals, truth)
  return(paste(capture.output(print(ans)), collapse = "\n"))
}

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
#memorial.day.effect <- c(.3, 3, .5)
memorial.day.effect <- c(10, 20, 30)
memorial.day.dates <- as.Date(c("2014-05-26", "2015-05-25"))
y <- AddHolidayEffect(y, memorial.day.dates, memorial.day.effect)

presidents.day <- NamedHoliday("PresidentsDay")
#presidents.day.effect <- c(.5, 2, .25)
presidents.day.effect <- c(40, 50, 60)
presidents.day.dates <- as.Date(c("2014-02-17", "2015-02-16"))
y <- AddHolidayEffect(y, presidents.day.dates, presidents.day.effect)

labor.day <- NamedHoliday("LaborDay")
#labor.day.effect <- c(1, 2, 1)
labor.day.effect <- c(70, 80, 90)
labor.day.dates <- as.Date(c("2014-09-01", "2015-09-07"))
y <- AddHolidayEffect(y, labor.day.dates, labor.day.effect)

## The holidays can be in any order.
holiday.list <- list(memorial.day, labor.day, presidents.day)
number.of.holidays <- length(holiday.list)

## In a real example you'd want more than 100 MCMC iterations.
niter <- 500

test_that("regression holiday model works", {
  ss <- AddLocalLevel(list(), y)
  ss <- AddRegressionHoliday(ss, y, holiday.list = holiday.list)
  model <- bsts(y, state.specification = ss, niter = niter, seed = 8675309, ping = niter)
  expect_that(model, is_a("bsts"))
  expect_that(model$MemorialDay, is_a("matrix"))
  expect_that(nrow(model$MemorialDay), equals(niter))
  expect_that(ncol(model$MemorialDay), equals(length(memorial.day.effect)))
  expect_true(CheckMcmcMatrix(model$MemorialDay, memorial.day.effect),
    info = McmcMatrixReport(model$MemorialDay, memorial.day.effect))
  expect_true(CheckMcmcMatrix(model$LaborDay, labor.day.effect),
    info = McmcMatrixReport(model$LaborDay, labor.day.effect))
  expect_true(CheckMcmcMatrix(model$PresidentsDay, presidents.day.effect),
    info = McmcMatrixReport(model$PresidentsDay, presidents.day.effect))
})

test_that("hierarchical model runs", {
  ## Try again with some shrinkage.  With only 3 holidays there won't be much
  ## shrinkage.
  ss2 <- AddLocalLevel(list(), y)
  ss2 <- AddHierarchicalRegressionHoliday(ss2, y, holiday.list = holiday.list)
  model2 <- bsts(y, state.specification = ss2, niter = niter, seed = 8675309, ping = niter)
  expect_that(model2, is_a("bsts"))
  expect_that(model2$holiday.coefficients, is_a("array"))
  expect_that(dim(model2$holiday.coefficients),
    equals(c(niter, number.of.holidays, 3)))
  }
)

test_that("random walk holiday works", {
  ss <- AddLocalLevel(list(), y)
  ss <- AddRandomWalkHoliday(ss, y, memorial.day)
  ss <- AddRandomWalkHoliday(ss, y, labor.day)
  ss <- AddRandomWalkHoliday(ss, y, presidents.day)
  model <- bsts(y, state.specification = ss, niter = niter, seed = 8675309, ping = niter)
  expect_that(model, is_a("bsts"))
  expect_that(length(dim(model$state.contributions)), equals(3))
  expect_true(memorial.day$name %in% dimnames(model$state.contributions)[[2]])
  expect_true(labor.day$name %in% dimnames(model$state.contributions)[[2]])
  expect_true(presidents.day$name %in% dimnames(model$state.contributions)[[2]])
})
