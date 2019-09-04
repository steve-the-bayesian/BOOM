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
  model <- bsts(train$iclaimsNSA ~ train$idaho.unemployment,
    state.specification = ss,
    niter = 100)

  test_subset <- cbind("department.of.unemployment" = test$department.of.unemployment)
  pred <- predict(model, newdata = test_subset)
})
