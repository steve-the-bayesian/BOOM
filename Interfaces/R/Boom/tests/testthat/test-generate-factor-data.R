library(Boom)
library(testthat)

context("test-generate-factor-data")

set.seed(8675309)

test_that("GenerateFactorData", {
  
  foo <- GenerateFactorData(list(a = c("foo", "bar", "baz"),
    b = c("larry", "moe", "curly", "shemp")),
    50)

  ## Check that we got a data frame back.
  expect_true(is.data.frame(foo))
  
  ## It should have 50 rows.
  expect_equal(50, nrow(foo))

  ## All variables must be factore.
  expect_true(all(sapply(foo, is.factor)))

  ## We should get all the levels, even if some of them don't appear
  ## because of random sampling.
  bar <- GenerateFactorData(list(a = c("foo", "bar", "baz"),
    b = c("larry", "moe", "curly", "shemp")),
    2)
  expect_equal(4, nlevels(bar[, 2]))
  expect_equal(3, nlevels(bar[, 1]))
})
