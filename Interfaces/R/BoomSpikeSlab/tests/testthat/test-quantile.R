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

test_that("quantile regression runs with Gaussian data", {
  set.seed(8675309)
  n <- 50
  x <- rnorm(n)
  y <- rnorm(n, 4 * x)
  model <- qreg.spike(y ~ x,
                      quantile = .8,
                      niter = 1000,
                      expected.model.size = 100,
                      seed = 8675309)
  ## Should get a slope near 4 and an intercept near qnorm(.8).
  ## PlotManyTs(foo$beta[-(1:100),], same.scale = T, truth = c(qnorm(.8), 4))

  expect_true(CheckMcmcMatrix(model$beta, truth = c(qnorm(.8), 4), burn = 100),
    info = McmcMatrixReport(model$beta, truth = c(qnorm(.8), 4), burn = 100))
})
