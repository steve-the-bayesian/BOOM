library(testthat)
library(Boom)

context("ToString")

test_that("ToString formats correctly", {
  m <- matrix(1:6, ncol = 2)
  printed.matrix <- ToString(m)
  expect_equal(printed.matrix,
       "     [,1] [,2]\n[1,]    1    4\n[2,]    2    5\n[3,]    3    6 \n")

  y <- c(1, 2, 3, 3, 3, 3, 3, 3)
  tab <- table(y)
  expect_equal(ToString(tab), "\nvalues:  1 2 3 \ncounts:  1 1 6  \n")
})
