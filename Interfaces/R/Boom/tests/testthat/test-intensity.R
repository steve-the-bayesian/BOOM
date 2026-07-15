library(Boom)
library(testthat)

context("test-intensity")

set.seed(8675309)

test_that("Intensity handles Date input", {
  dates <- as.Date("2020-01-01") + sample(0:364, 200, replace = TRUE)
  intensity <- Intensity(dates)
  expect_s3_class(intensity, "Intensity")
  expect_s3_class(intensity$x, "Date")
  ## The intensity integrates to the number of events rather than to 1.
  area <- sum(diff(as.numeric(intensity$x)) *
              (head(intensity$y, -1) + tail(intensity$y, -1)) / 2)
  expect_equal(area, length(dates), tolerance = 0.05 * length(dates))
})

test_that("Intensity handles POSIXt input", {
  times <- as.POSIXct("2020-01-01 00:00:00", tz = "UTC") +
    sample(0:(86400 * 30), 200, replace = TRUE)
  intensity <- Intensity(times)
  expect_s3_class(intensity, "Intensity")
  expect_s3_class(intensity$x, "POSIXct")
})

test_that("Intensity rejects unsupported classes", {
  expect_error(Intensity(1:10),
               "Intensity only supports POSIXt, Date, and yearmon classes.")
})

test_that("Intensity handles yearmon input", {
  skip_if_not_installed("zoo")
  months <- zoo::as.yearmon("2020-01") + sample(0:23, 200, replace = TRUE) / 12
  intensity <- Intensity(months)
  expect_s3_class(intensity, "Intensity")
  expect_s3_class(intensity$x, "yearmon")
  ## The plot method should run without error on the yearmon axis.  Route the
  ## output to a temporary device so the test leaves no files behind.
  pdf(tempfile(fileext = ".pdf"))
  on.exit(dev.off(), add = TRUE)
  expect_silent(plot(intensity))
})
