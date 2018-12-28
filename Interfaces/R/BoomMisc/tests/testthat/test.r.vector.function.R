library(BoomMisc)
library(testthat)

context("r.vector.function")

set.seed(8675309)

testfun <- function(x, y) { return(x + y) }
test_that("function with dots evaluates correctly.", {
  result <- .EvaluateFunction(tesfun, x = 3, y = 5)
  expect_equal(result, 8)
})

othertestfun <- function(x) { return(x + 3) }
test_that("function without dots evaluates correctly.", {
  expect_equal(.EvaluateFunction(othertestfun, 7), 10)
})
