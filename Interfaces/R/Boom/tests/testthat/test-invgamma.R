set.seed(8675309)
library(testthat)
library(Boom)

test_that("rinvgamma", {
  ndraws <- 100000
  draws <- rinvgamma(ndraws, 70, 4);
  gamma.draws <- rgamma(ndraws, 70, 4)
  test.result <- ks.test(1/draws, gamma.draws)
  expect_true(test.result$p.value > .05)
})

test_that("dinvgamma", {
  integral <- integrate(dgamma, lower = 0, upper = Inf, shape = 4, rate = 17)
  expect_true((integral$value - 1) < integral$abs.error)
  expect_true(integral$abs.error < 1e-5)
})

test_that("pinvgamma and qinvgamma", {
  deviates <- rinvgamma(10, 3, 7)
  probs <- pinvgamma(deviates, 3, 7)
  complementary.probs <- pinvgamma(deviates, 3, 7, lower.tail = FALSE)
  expect_true(all(abs(complementary.probs + probs - 1.0) < 1e-7))
  expect_true(all(probs >= 0))
  expect_true(all(complementary.probs >= 0))

  log.probs <- pinvgamma(deviates, 3, 7, log = TRUE)
  expect_true(all(abs(log.probs - log(probs)) < 1e-7))
  log.complementary.probs <- pinvgamma(deviates, 3, 7, log = TRUE,
                                       lower.tail = FALSE)
  expect_true(all(abs(log.complementary.probs - log(complementary.probs)) < 1e-7))

  same.deviates <- qinvgamma(probs, 3, 7)
  expect_true(all(abs(deviates - same.deviates) < 1e-7))

  same.deviates2 <- qinvgamma(complementary.probs, 3, 7, lower.tail = FALSE)
  expect_true(all(abs(deviates - same.deviates2) < 1e-7))

  same.deviates3 <- qinvgamma(log.probs, 3, 7, log = TRUE)
  expect_true(all(abs(deviates - same.deviates3) < 1e-7))

  same.deviates4 <- qinvgamma(log.complementary.probs, 3, 7,
                              log = TRUE, lower.tail = FALSE)
  expect_true(all(abs(deviates - same.deviates4) < 1e-7))
})
