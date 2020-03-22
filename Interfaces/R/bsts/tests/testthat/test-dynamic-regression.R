
set.seed(8675309)

library(bsts)

time.dimension <- 100
xdim <- 3
beta.sd <- c(.1, .2, .05)
residual.sd <- .7

beta <- matrix(nrow = xdim, ncol = time.dimension)
beta[, 1] <- rnorm(xdim)
for (i in 2:time.dimension) {
  beta[, i] <- rnorm(xdim, beta[, i - 1], beta.sd)
}

predictors <- matrix(rnorm(time.dimension * xdim),
  nrow = time.dimension, ncol = xdim)
yhat <- rowSums(predictors * t(beta))
y <- rnorm(time.dimension, yhat, residual.sd)

## Check that the model runs with a default prior.
test_that("model runs with default prior", {
  ss <- AddDynamicRegression(list(), y ~ predictors)
  model <- bsts(y, state.specification = ss, niter = 100)

  ## Check that you can specify separate priors.
  ss <- AddDynamicRegression(list(), y ~ predictors,
    model.options = DynamicRegressionRandomWalkOptions(
      sigma.prior = list(
        SdPrior(beta.sd[1], 10),
        SdPrior(beta.sd[2], 10),
        SdPrior(beta.sd[3], 10))))
  model <- bsts(y, state.specification = ss, niter = 1000, seed = 8675309)
  burn <- SuggestBurn(.1, model)
  CheckMcmcMatrix(model$dynamic.regression.coefficients[, 1, ],
    beta[1, ], burn = burn)
  CheckMcmcMatrix(model$dynamic.regression.coefficients[, 2, ],
    beta[2, ], burn = burn)
  CheckMcmcMatrix(model$dynamic.regression.coefficients[, 3, ],
    beta[3, ], burn = burn)

  ## Check that you can specify a single prior.
  ss <- AddDynamicRegression(list(), y ~ predictors,
    model.options = DynamicRegressionRandomWalkOptions(
      sigma.prior = SdPrior(beta.sd[1], 1)))
  model <- bsts(y, state.specification = ss, niter = 100)
})

test_that("predict method runs without crashing for DLM's", {
  library(bsts)
  ### Load the data
  data(iclaims)
  train <- window(initial.claims, start = "2004-01-04", end="2010-01-01")
  test <- window(initial.claims, start="2010-01-02")

  # Create model
  ss <- AddLocalLinearTrend(list(), train$iclaimsNSA)
  ss <- AddSeasonal(ss, train$iclaimsNSA, nseasons = 52)
  # Dynamic regression component
  ss <- AddDynamicRegression(ss, formula = iclaimsNSA ~ unemployment.office,
    data = train)

  # Train it
  model <- bsts(train$iclaimsNSA, state.specification = ss, niter = 1000)
  test_subset <- cbind(
    "department.of.unemployment" = test$department.of.unemployment,
    "unemployment.office" = test$unemployment.office)
  pred <- predict(model, newdata = test_subset)
})

test_that("predict method runs without crashing for DLM's with static regressors", {
  library(bsts)
  ### Load the data
  data(iclaims)
  train <- window(initial.claims, start = "2004-01-04", end="2010-01-01")
  test <- window(initial.claims, start="2010-01-02")

  # Create model
  ss <- AddLocalLinearTrend(list(), train$iclaimsNSA)
  ss <- AddSeasonal(ss, train$iclaimsNSA, nseasons = 52)
  # Dynamic regression component
  ss <- AddDynamicRegression(ss,
    formula = iclaimsNSA ~ unemployment.office,
    data = train)

  # Train it
  model <- bsts(iclaimsNSA ~ idaho.unemployment,
    state.specification = ss,
    niter = 100,
    data = train)

  test.subset <- cbind(test,
    "department.of.unemployment" = test$department.of.unemployment)
  #  pred <- predict(model, newdata = test.subset)
  pred <- predict(model, newdata = test)

})

test_that("dynamic regression fails gracefully with non-trivial time stamps", {
  library(bsts)
  ## From AddDynamicRegression example
  set.seed(8675309)
  n <- 1000
  x <- matrix(rnorm(n))

  # beta follows a random walk with sd = .1 starting at -12.
  beta <- cumsum(rnorm(n, 0, .1)) - 12

  # level is a local level model with sd = 1 starting at 18.
  level <- cumsum(rnorm(n)) + 18

  # sigma.obs is .1
  error <- rnorm(n, 0, .1)

  y <- level + x * beta + error

  ss <- list()
  ss <- AddLocalLevel(ss, y)
  ss <- AddDynamicRegression(ss, y ~ x)

  ## This works:
  model <- bsts(y, state.specification = ss, niter = 100, seed = 8675309)

  ## This fails and crashes R:
  new_timestamps <- sort(sample(1:2000, 1000))
  expect_error(
    model <- bsts(y, state.specification = ss, niter = 100, seed = 8675309,
      timestamps = new_timestamps),
    "Dynamic regression models are only supported with trivial time stamps.")

})
