holiday <- NamedHoliday("MemorialDay", days.before = 2, days.after = 2)
timestamps <- seq.Date(from = as.Date("2001-01-01"), by = "day",
   length.out = 365 * 10)

influence <- DateRange(holiday, timestamps)

test_that("DateRange returns a two-column data frame of dates", {
  expect_that(influence, is_a("data.frame"))
  expect_that(ncol(influence), equals(2))
  expect_that(influence[, 1], is_a("Date"))
  expect_that(influence[, 2], is_a("Date"))
  expect_true(all(influence[, 1] <= influence[, 2]))
  })
