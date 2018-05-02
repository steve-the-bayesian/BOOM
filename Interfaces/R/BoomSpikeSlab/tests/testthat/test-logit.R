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
library(MASS)

test_that("Calls with the same random seed return the same draws", {
  n <- 100
  niter <- 1000
  x <- rnorm(n)
  beta0 <- -3
  beta1 <- 1
  logits <- beta0 + beta1*x
  probabilities <- plogis(logits)
  y <- as.numeric(runif(n) <= probabilities)

  ## Do the check with model selection turned off, signaled by
  ## expected.model.size > number of variables.
  model1 <- logit.spike(y ~ x, niter = niter, expected.model.size = 3,
                        seed = 31417, ping = -1)
  model2 <- logit.spike(y ~ x, niter = niter, expected.model.size = 3,
                        seed = 31417, ping = -1)
  expect_equal(model1$beta, model2$beta)

  ## Now do the check with model selection turned on.
  model1 <- logit.spike(y ~ x, niter = niter, seed = 31417, ping = -1)
  model2 <- logit.spike(y ~ x, niter = niter, seed = 31417, ping = -1)
  expect_equal(model1$beta, model2$beta)
})

test_that("Pima Indians works without threads", {
  if (requireNamespace("MASS")) {
    data(Pima.tr, package = "MASS")
    data(Pima.te, package = "MASS")
    pima <- rbind(Pima.tr, Pima.te)
    niter <- 1000
    sample.size <- nrow(pima)
    m <- logit.spike(type == "Yes" ~ ., data = pima, niter = niter)
    expect_true(!is.null(m$beta))
    expect_true(is.matrix(m$beta))
    expect_equal(dim(m$beta), c(1000, 8))
    expect_equal(length(m$fitted.probabilities), sample.size)
    expect_equal(length(m$fitted.logits), sample.size)
    expect_equal(length(m$deviance.residuals), sample.size)
    expect_equal(length(m$log.likelihood), niter)
    expect_true(!is.null(m$prior))

    m.summary <- summary(m)
    expect_true(!is.null(m.summary$coefficients))
    expect_true(is.matrix(m.summary$predicted.vs.actual))
    expect_equal(ncol(m.summary$predicted.vs.actual), 2)
    expect_equal(nrow(m.summary$predicted.vs.actual), 10)
    expect_equal(length(m.summary$deviance.r2.distribution), 1000)
    expect_equal(length(m.summary$deviance.r2), 1)
    expect_true(is.numeric(m.summary$deviance.r2))

    # The following is always true regardless of the data.
    expect_true(m.summary$max.log.likelihood >= m.summary$mean.log.likelihood)

    # The following is true for this data.
    expect_true(m.summary$max.log.likelihood > m.summary$null.log.likelihood)

    m <- logit.spike(type == "Yes" ~ ., data = Pima.tr, niter = niter)
    predictions <- predict(m, newdata = Pima.te)
  }
})

test_that("Pima indians analysis runs with threads", {
  if (requireNamespace("MASS")) {
    data(Pima.tr, package = "MASS")
    data(Pima.te, package = "MASS")
    pima <- rbind(Pima.tr, Pima.te)
    niter <- 1000
    sample.size <- nrow(pima)
    m <- logit.spike(type == "Yes" ~ ., data = pima,
                     niter = niter, nthreads = 8)
    expect_true(!is.null(m$beta))
    expect_true(is.matrix(m$beta))
    expect_equal(dim(m$beta), c(1000, 8))
    expect_equal(length(m$fitted.probabilities), sample.size)
    expect_equal(length(m$fitted.logits), sample.size)
    expect_equal(length(m$deviance.residuals), sample.size)
    expect_equal(length(m$log.likelihood), niter)
    expect_true(!is.null(m$prior))

    m.summary <- summary(m)
    expect_true(!is.null(m.summary$coefficients))
    expect_true(is.matrix(m.summary$predicted.vs.actual))
    expect_equal(ncol(m.summary$predicted.vs.actual), 2)
    expect_equal(nrow(m.summary$predicted.vs.actual), 10)
    expect_equal(length(m.summary$deviance.r2.distribution), 1000)
    expect_equal(length(m.summary$deviance.r2), 1)
    expect_true(is.numeric(m.summary$deviance.r2))

    # The following is always true regardless of the data.
    expect_true(m.summary$max.log.likelihood >= m.summary$mean.log.likelihood)

    # The following is true for this data.
    expect_true(m.summary$max.log.likelihood > m.summary$null.log.likelihood)

    tmp.file <- tempfile()
    png(tmp.file)
    plot(m)
    plot(m, "coefficients")
    plot(m, "scaled.coefficients")
    plot(m, "fit")
    plot(m, "residuals")
    plot(m, "size")
    dev.off()
}
})

test_that("Small number of cases", {
  ## The model should run and give sensible results even if,
  ## e.g. there are no/all successes or there are very few data
  ## points.  This code should produce a warning about the
  ## prior.success.probability begin zero.

  x <- matrix(rnorm(45), ncol = 9)
  x <- cbind(1, x)
  y <- rep(FALSE, nrow(x))

  expect_warning(
      m <- logit.spike(y ~ x, niter = 100),
      "Fudging around the fact that prior.success.probability is zero. Setting it to 0.14286")
  invisible(NULL)
})

test_that("contrasts.arg respected", {
  if (requireNamespace("MASS")) {
    data(Pima.tr, package = "MASS")
    data(Pima.te, package = "MASS")
    pima <- rbind(Pima.tr, Pima.te)
    df0 <- pima[1:100, ]
    df1 <- pima[101:532, ]
    xlevels <- list(skin = c("10", "20", "30", "40", "50", "60", "100"))
    df1$skin <- factor(round(df1$skin, -1), levels = xlevels$skin)
    df0$skin <- factor(round(df0$skin, -1))

    model1 <- logit.spike(type == "Yes" ~ ., drop.unused.levels = FALSE,
                          data = df1, niter = 500, seed = 31417, ping = -1)
    model2 <- logit.spike(type == "Yes" ~ ., drop.unused.levels = FALSE,
                          data = df1, niter = 500, seed = 31417, ping = -1,
                          contrasts = list(skin="contr.helmert"))
    model3 <- logit.spike(type == "Yes" ~ ., drop.unused.levels = FALSE,
                          data = df1, niter = 500, seed = 31417, ping = -1,
                          contrasts = list(skin="contr.poly"))
    model4 <- logit.spike(type == "Yes" ~ ., drop.unused.levels = FALSE,
                          data = df1, niter = 500, seed = 31417, ping = -1,
                          contrasts = list(skin="contr.sum"))
    model5 <- logit.spike(type == "Yes" ~ ., drop.unused.levels = FALSE,
                          data = df1, niter = 500, seed = 31417, ping = -1,
                          contrasts = list(skin="contr.SAS"))

    expect_equal(model1$xlevels, xlevels)
    expect_equal(model2$xlevels, xlevels)
    expect_equal(model3$xlevels, xlevels)
    expect_equal(model4$xlevels, xlevels)
    expect_equal(model5$xlevels, xlevels)

    betanames <- c("(Intercept)", "npreg", "glu", "bp", "skin20", "skin30",
                   "skin40", "skin50", "skin60", "skin100", "bmi", "ped", "age")
    expect_equal(ncol(model1$beta), length(betanames))
    expect_equal(ncol(model2$beta), length(betanames))
    expect_equal(ncol(model3$beta), length(betanames))
    expect_equal(ncol(model4$beta), length(betanames))
    expect_equal(ncol(model5$beta), length(betanames))
  }
})
