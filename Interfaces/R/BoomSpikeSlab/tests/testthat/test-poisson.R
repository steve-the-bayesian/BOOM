# Copyright 2018 Google LLC. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

library(BoomSpikeSlab)
library(testthat)

test_that("Poisson without model selection matches MLE", {
## This test checks the behavior of the Poisson regression without
## model selection.  It compares the posterior means and SD's to the
## MLE and asymptotic standard errors.  It also checks that the

seed <- as.integer(8675309)
set.seed(seed)

cat("test-poisson\n")

n <- 100
  number.of.variables <- 4
  design <- matrix(rnorm(n * number.of.variables), nrow = n)
  design[, 1] <- 1
  beta <- rnorm(number.of.variables)
  eta <- design %*% beta
  exposure <- rgamma(n, 1, 1)
  lambda <- exp(eta)
  y <- rpois(n, lambda * exposure)
  data <- as.data.frame(cbind(y, design[, -1]))
  model <- poisson.spike(
      y ~ .,
      exposure = exposure,
      data = data,
      niter = 1000,
      expected.model.size = 5,
      ping = -1,
      seed = seed)
  burn <- 100
  beta.draws <- model$beta[-(1:burn),]
  beta.limits <- apply(beta.draws, 2, quantile, probs = c(0, 1))
  expect_true(all(beta.limits[1, ] < beta),
    info = paste(
      "At least one element of the true beta was missed",
      "by the lower endpoint of its posterior sample."))
  expect_true(all(beta.limits[2, ] > beta),
            info = paste(
                "At least one element of the true beta was missed",
                "by the upper endpoint of its posterior sample."))

  beta.mean <- colMeans(beta.draws)
  beta.posterior.sd <- apply(beta.draws, 2, sd)
  mle <- glm(y ~ ., data = data, offset = log(exposure), family = poisson())
  beta.table <- summary(mle)$coef
  beta.hat <- beta.table[, 1]
  expect_true(all(abs((beta.mean - beta.hat) / beta.hat) < .1))
  beta.se <- beta.table[, 2]
  expect_true(all(abs(beta.posterior.sd - beta.se) / beta.se < .15))
})

test_that("PoissonRegression finds right variables", {
  seed <- as.integer(8675309)
  set.seed(seed)
  n <- 500
  number.of.variables <- 20
  design <- matrix(rnorm(n * number.of.variables), nrow = n)
  design[, 1] <- 1
  beta <- rep(0, number.of.variables)
  included <- rep(FALSE, number.of.variables)
  included[c(1, 3, 7)] <- TRUE
  beta[included] <- c(-2, 1, 4)
  eta <- design %*% beta
  exposure <- rgamma(n, 1, 1)
  lambda <- exp(eta)
  y <- rpois(n, lambda * exposure)
  my.data <- as.data.frame(cbind(y, design[, -1]))
  model <- poisson.spike(
      y ~ .,
      exposure = exposure,
      data = my.data,
      niter = 1000,
      expected.model.size = 3,
      ping = -1,
      seed = seed)
  burn <- 100
  beta.draws <- model$beta[-(1:burn), ]
  inclusion.probabilities <- colMeans(beta.draws != 0)
  expect_true(all(inclusion.probabilities[included] > .5))
  expect_true(all(inclusion.probabilities[!included] < .5))

  included.data <- as.data.frame(cbind(y, design[, included][, -1]))
  mle <- glm(y ~ .,
             data = included.data,
             offset = log(exposure),
             family = poisson())
  ## Trimming burn-in is important here because of the model
  ## selection.
  beta.mean <- colMeans(beta.draws[, included])
  beta.posterior.sd <- apply(beta.draws[, included], 2, sd)
  beta.table <- summary(mle)$coef
  beta.hat <- beta.table[, 1]
  expect_true(all(abs((beta.mean - beta.hat) / beta.hat) < .1),
    info = "Posterior mean of beta is far from beta.hat.")
  beta.se <- beta.table[, 2]
  expect_true(all(abs(beta.posterior.sd - beta.se) / beta.se < .1),
    info = "Posterior sd of beta is far from SE(beta.hat).")

  pred <- predict(model, newdata = my.data)

  tmp.file <- tempfile()
  png(tmp.file)
  plot(model)
  plot(model, "size")
  plot(model, "coefficients")
  plot(model, "scaled.coefficients")
  dev.off()
})
