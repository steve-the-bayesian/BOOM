library(bsts)
library(testthat)

# GDP figures for 57 countries as reported by the OECD.
data(gdp)
series.id <- gdp$Country
timestamps <- gdp$Time

test_that("Wide long conversion", {
  wide <- matrix(rnorm(40), nrow = 10, ncol = 4)
  long <- WideToLong(wide)
  wide2 <- LongToWide(long$values, long$series, long$time)
  expect_equal(sum((wide-wide2)^2), 0.0)

  wide.with.na <- wide
  wide.with.na[c(1, 3, 7), c(2, 3)] <- NA
  long.with.na <- WideToLong(wide.with.na)
  wide.with.na2 <- LongToWide(long.with.na$values, long.with.na$series,
    long.with.na$time)
})



