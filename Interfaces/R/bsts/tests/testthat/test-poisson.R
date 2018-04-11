library(bsts)
library(testthat)

data(shark)
test_that("Poisson bsts", {

  logshark <- log1p(shark$Attacks)
  
  ss.level <- AddLocalLevel(list(), y = logshark)
  model.level <- bsts(shark$Attacks, ss.level, niter = 1000, family = "poisson", seed = 8675309)
  expect_that(model, is_a("bsts"))

  ss.level <- AddLocalLevel(list(), y = logshark)
  model.level <- bsts(cbind(shark$Attacks, shark$Population / 1000),
    state.specification = ss.level, niter = 1000, family = "poisson", seed = 8675309)

  ## This currently does not work, because one or more of the states moves into
  ## very negative territory and can't recover.  Not sure why that is a problem
  ## for LLT and not local level or semilocal.
  ss <- AddLocalLinearTrend(list(), y = logshark)
  model <- bsts(shark$Attacks, ss, niter = 1000, family = "poisson",
    model.options = BstsOptions(save.full.state = TRUE))
  expect_that(model, is_a("bsts"))
  expect_true(all(abs(model$state.contributions) < 1e+10))

  ss <- AddLocalLinearTrend(list(), logshark,
    initial.level.prior = NormalPrior(0, .1),
    initial.slope.prior = NormalPrior(.16, .1))
  model <- bsts(shark$Attacks, ss, niter = 1000, ping = 500, family = "poisson",
    model.options = BstsOptions(save.full.state = TRUE))
  
  ss.semi <- AddSemilocalLinearTrend(list(), y = logshark)
  model.semi <- bsts(shark$Attacks, ss.semi, niter = 1000, family = "poisson")

  ss.student <- AddStudentLocalLinearTrend(list(), y = logshark)
  model.student <- bsts(shark$Attacks, ss.student, niter = 1000, family = "poisson")
})
