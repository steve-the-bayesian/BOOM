
context("mvn")

test_that("dmvn", {
  mu <- c(-2, 0, 7)
  sigma <- c(2, 5, 3)
  y <- rnorm(3, mean = mu, sd = sigma)

  Sigma.matrix <- diag(sigma^2)
  expect_true(abs(sum(dnorm(y, mu, sigma, log = TRUE))
                - dmvn(y, mu, Sigma.matrix, log = TRUE))
            < 1e-4)

  siginv <- solve(Sigma.matrix)
  ldsi <- log(det(siginv))

  f1 <- dmvn(y, mu, Sigma.matrix, log = TRUE)
  f2 <- dmvn(y, mu, siginv = siginv, ldsi = ldsi, log = TRUE)
  expect_equal(f1, f2)
  f3 <- dmvn(y, mu, siginv = siginv, ldsi = NULL, log = TRUE)
  expect_equal(f2, f3)

  V <- rWishart(10, Sigma.matrix)
  V.inv <- solve(V)
  log.determinant <- log(det(V.inv))
  f4 <- dmvn(y, mu, V, log = TRUE)
  f5 <- dmvn(y, mu, siginv = V.inv, log = TRUE)
  f6 <- dmvn(y, mu, siginv = V.inv, ldsi = log.determinant, log = TRUE)
  expect_equal(f4, f5)
  expect_equal(f5, f6)

  Y <- rbind(y, y, y, y, y)
  f7 <- dmvn(Y, mu, V, log = TRUE)
  f8 <- dmvn(Y, mu, siginv = V.inv, log = TRUE)
  f9 <- dmvn(Y, mu, siginv = V.inv, ldsi = log.determinant, log = TRUE)
  expect_equal(5, length(f7))
  expect_equal(f7, f8)
  expect_equal(f8, f9)


  y <- rnorm(4)
  density <- dmvn(y, rep(0, 4), diag(rep(1, 4)), log = TRUE)
  density.direct <- sum(dnorm(y, log = TRUE))
  expect_equal(density.direct, density)

  Sigma <- crossprod(matrix(rnorm(16), ncol = 4))
  mu <- rnorm(4)
  y <- as.numeric(t(chol(Sigma)) %*% y + mu)

  density <- dmvn(y, mu, Sigma, log = TRUE)
  p <- length(y)
  qform <- mahalanobis(y, mu, Sigma, inverted = FALSE)

  density.direct <- log(1 / sqrt(2 * pi)^p) - .5 * log(det(Sigma)) - .5 * qform
  expect_equal(density.direct, density)
})

test_that("rmvn", {
  set.seed(8675309)
  mu <- c(-2, 0, 7)
  Sigma <- matrix(c(8, 6, 7,
                    6, 9, 5,
                    7, 5, 30), ncol = 3)
  y <- rmvn(1e+5, mu, Sigma)

  ybar <- colMeans(y)
  expect_true(all(abs(ybar - mu) < .01))

  S <- var(y)
  expect_true(max(abs(S - Sigma)) < .2)
})
