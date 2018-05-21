library(Boom)
library(testthat)
context("MatchDataFrame")

test_that("rows only", {
  x1 <- data.frame(larry = rnorm(10), moe = 1:10, curly = rpois(10, 2))
  x2 <- x1[c(1:5, 10:6), ]
  m <- MatchDataFrame(x1, x2)
  expect_true(all(abs(m$column.permutation - 1:3) < 1e-7))
  expect_true(all(x2[m$row.permutation,
                   m$column.permutation] == x1))
  expect_true(all(x2[m$row.permutation, ] == x1))
})

test_that("columns only", {
  x1 <- data.frame(larry = rnorm(10), moe = 1:10, curly = rpois(10, 2))
  x2 <- x1[, c(3, 1, 2)]
  m <- MatchDataFrame(x1, x2)
  expect_true(all(abs(m$row.permutation - 1:10) < 1e-7))
  expect_true(all(x2[m$row.permutation,
                   m$column.permutation] == x1))
  expect_true(all(x2[, m$column.permutation] == x1))
})

test_that("rows and columns", {
  x1 <- data.frame(larry = rnorm(10), moe = 1:10, curly = rpois(10, 2))
  x2 <- x1[c(1:5, 10:6), c(3, 1, 2)]
  m <- MatchDataFrame(x1, x2)
  expect_true(all(x2[m$row.permutation,
                   m$column.permutation] == x1))
})

test_that("no match", {
  expect_error(
      MatchDataFrame(data.frame(x = 1:3, y = 5:7),
                     data.frame(y = 6:8, x = 1:3)),
      regexp = "Not all rows could be matched.")


  expect_error(
      MatchDataFrame(data.frame(x = 1:3, y = 5:7),
                     data.frame(z = 6:8, x = 1:3)),
      regexp = "Not all columns could be matched.")

  expect_error(
      MatchDataFrame(data.frame(x = 1:3, y = 5:7),
                     data.frame(y = 6:9, x = 1:4)),
      regexp = "Data are of different sizes")
})
