library(bsts)
library(testthat)

data(shark)
logshark <- log1p(shark$Attacks)
seed <- 8675309

test_that("Poisson bsts", {

  ss.level <- AddLocalLevel(list(), y = logshark)
  model.level <- bsts(shark$Attacks, ss.level, niter = 500,
    ping = 250, family = "poisson", seed = seed)
  expect_that(model.level, is_a("bsts"))
  expect_true(all(abs(model.level$state.contributions) < 10))

  ss.level <- AddLocalLevel(list(), y = logshark)
  model.level <- bsts(cbind(shark$Attacks, shark$Population / 1000),
    state.specification = ss.level, niter = 500, family = "poisson",
    ping = 250, seed = seed)
  expect_true(all(abs(model.level$state.contributions) < 15))

  ss <- AddLocalLinearTrend(list(), y = logshark)
  model <- bsts(shark$Attacks, ss, niter = 500, family = "poisson", ping = 250,
    seed = seed)
  expect_that(model, is_a("bsts"))

  expect_true(all(abs(model$state.contributions) < 10))

  ss <- AddLocalLinearTrend(list(), logshark,
    initial.level.prior = NormalPrior(0, .1),
    initial.slope.prior = NormalPrior(.16, .1))
  model <- bsts(shark$Attacks, ss, niter = 500, ping = 250,
    family = "poisson", seed = seed)
  expect_that(model, is_a("bsts"))
  expect_true(all(abs(model$state.contributions) < 10))

  ss.semi <- AddSemilocalLinearTrend(list(), y = logshark)
  model.semi <- bsts(shark$Attacks, ss.semi, niter = 500,
    ping = 250, family = "poisson", seed = seed)
  expect_that(model.semi, is_a("bsts"))
  expect_true(all(abs(model.semi$state.contributions) < 10))

  ss.student <- AddStudentLocalLinearTrend(list(), y = logshark)
  model.student <- bsts(shark$Attacks, ss.student, niter = 500,
    ping = 250, family = "poisson", seed = seed)
  expect_that(model.student, is_a("bsts"))
  expect_true(all(abs(model.student$state.contributions) < 10))

  ## Add an unrelated predictor.
  shark$x <- rnorm(nrow(shark))
  shark.training <- shark[1:48,]
  shark.test <- shark[49:54, ]
  ss.reg <- AddLocalLinearTrend(list(), y = logshark)
  ss.reg <- AddDynamicRegression(ss.reg, log1p(Attacks) ~ x, data = shark)
  model <- bsts(cbind(shark.training$Attacks, shark.training$Population / 1000),
    ss.reg, niter = 500, family = "poisson", seed = seed)
  pred <- predict(model, newdata = shark.test, trials.or.exposure = max(model$exposure))
  expect_that(pred, is_a("bsts.prediction"))
  expect_equal(ncol(pred$distribution), 6)
})
