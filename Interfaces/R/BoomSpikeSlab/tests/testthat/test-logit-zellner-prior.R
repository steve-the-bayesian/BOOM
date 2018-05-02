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


## In some sense LogitZellnerPrior is tested implicitly by tests for
## logit.spike and other clients that use it.  This test verifies its
## other behavior.

sample.size <- 20
xdim <- 4
predictors <- cbind(1, matrix(rnorm(sample.size * (xdim - 1)),
                              ncol = (xdim-1)))
trials <- 1 + rpois(sample.size, 1.3)
probs <- runif(sample.size)
successes <- rbinom(sample.size, trials, probs)
p.hat <- sum(successes) / sum(trials)

ComputePrecision <- function(predictors,
                             weight,
                             prior.information.weight,
                             diagonal.shrinkage) {
  V0 <- prior.information.weight * crossprod(sqrt(weight) * predictors) /
      nrow(predictors)
  diag.V0 <- diag(diag(V0))
  return(diagonal.shrinkage * diag.V0 + (1 - diagonal.shrinkage) * V0)
}

test_that("LogitZellnerPrior basic info", {
  prior <- LogitZellnerPrior(predictors = predictors,
                             successes = successes,
                             trials = trials)
  expect_equal(prior$prior.inclusion.probabilities,
    c(.25, .25, .25, .25))
  expect_equal(prior$mu, c(qlogis(p.hat), 0, 0, 0))
  ## The default prior.information.weight for LogitZellnerPrior is .01
  prior.information.weight <- .01
  diagonal.shrinkage <- .5
  weight <- p.hat * (1 - p.hat)
  expected.precision <- ComputePrecision(
      predictors, weight, prior.information.weight, diagonal.shrinkage)
  expect_equal(prior$siginv, expected.precision)
})

test_that("inclusion probabilities respected", {
  ## Check that prior inclusion probabilities are respected, and that
  ## they trump expected.model.size.
  prior.inclusion.probabilities <- c(1, 1, 0, .3)
  prior <- LogitZellnerPrior(
      predictors = predictors,
      successes = successes,
      trials = trials,
      expected.model.size = 2,
      prior.inclusion.probabilities = prior.inclusion.probabilities)
  expect_equal(prior.inclusion.probabilities,
    prior$prior.inclusion.probabilities)
})

test_that("prior.success.probability is used", {
## Check that successes and trials are not needed if
## prior.success.probability is included.
  prior.success.probability <- .25
  prior <- LogitZellnerPrior(
      predictors,
      prior.success.probability = prior.success.probability)
  prior.information.weight <- 0.01
  diagonal.shrinkage <- 0.5
  weight <- prior.success.probability * (1 - prior.success.probability)
  expected.precision <- ComputePrecision(
      predictors, weight, prior.information.weight, diagonal.shrinkage)
  expect_equal(expected.precision, prior$siginv)
})


test_that("Huge expected.model.size includes everything", {
  ## Check that huge values of expected.model.size cause everything to
  ## be included.
  prior <- LogitZellnerPrior(predictors, successes, trials,
                             expected.model.size = 50)
  expect_equal(prior$prior.inclusion.probabilities, rep(1, 4))
})

test_that("diagonal.shrinkage shrinks to the diagonal", {
  ## Check that diagonal.shrinkage shrinks towards the diagonal.
  prior <- LogitZellnerPrior(predictors, successes, trials,
                             diagonal.shrinkage = 1.0)
  lower.triangle <- prior$siginv[lower.tri(prior$siginv)]
  expect_true(all(lower.triangle == 0.0))

  upper.triangle <- prior$siginv[upper.tri(prior$siginv)]
  expect_true(all(upper.triangle == 0.0))

  expect_true(all(diag(prior$siginv) > 0))
})

test_that("optional.coefficient.estimate is used", {
  ## Check that the optional.coefficient.estimate is used.
  prior <- LogitZellnerPrior(
      predictors, successes, trials,
      optional.coefficient.estimate = 1:4)
  expect_equal(prior$mu, 1:4)
})

test_that("max.flips gets included", {
  ## Check max flips
  prior <- LogitZellnerPrior(predictors, successes, trials, max.flips = 17)
  expect_equal(prior$max.flips, 17)
})

