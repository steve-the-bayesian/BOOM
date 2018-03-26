TestDmvn <- function() {
  mu <- c(-2, 0, 7)
  sigma <- c(2, 5, 3)
  y <- rnorm(3, mean = mu, sd = sigma)

  Sigma.matrix <- diag(sigma^2)
  checkTrue(abs(sum(dnorm(y, mu, sigma, log = TRUE))
                - dmvn(y, mu, Sigma.matrix, log = TRUE))
            < 1e-4)

  siginv <- solve(Sigma.matrix)
  ldsi <- log(det(siginv))

  f1 <- dmvn(y, mu, Sigma.matrix, log = TRUE)
  f2 <- dmvn(y, mu, siginv = siginv, ldsi = ldsi, log = TRUE)
  checkEqualsNumeric(f1, f2)
  f3 <- dmvn(y, mu, siginv = siginv, ldsi = NULL, log = TRUE)
  checkEqualsNumeric(f2, f3)

  V <- rWishart(10, Sigma.matrix)
  V.inv <- solve(V)
  log.determinant <- log(det(V.inv))
  f4 <- dmvn(y, mu, V, log = TRUE)
  f5 <- dmvn(y, mu, siginv = V.inv, log = TRUE)
  f6 <- dmvn(y, mu, siginv = V.inv, ldsi = log.determinant, log = TRUE)
  checkEqualsNumeric(f4, f5)
  checkEqualsNumeric(f5, f6)

  Y <- rbind(y, y, y, y, y)
  f7 <- dmvn(Y, mu, V, log = TRUE)
  f8 <- dmvn(Y, mu, siginv = V.inv, log = TRUE)
  f9 <- dmvn(Y, mu, siginv = V.inv, ldsi = log.determinant, log = TRUE)
  checkEquals(5, length(f7))
  checkEqualsNumeric(f7, f8)
  checkEqualsNumeric(f8, f9)
}

TestRmvn <- function() {
  set.seed(8675309)
  mu <- c(-2, 0, 7)
  Sigma <- matrix(c(8, 6, 7,
                    6, 9, 5,
                    7, 5, 30), ncol = 3)
  y <- rmvn(1e+5, mu, Sigma)

  ybar <- colMeans(y)
  checkTrue(all(abs(ybar - mu) < .01))

  S <- var(y)
  checkTrue(max(abs(S - Sigma)) < .2)
}
