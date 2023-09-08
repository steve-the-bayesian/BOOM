GaussianSuf <- function(y) {
  ## Sufficient statistics for a Gaussian distribution given data 'y'.
  stopifnot(is.numeric(y))
  ans <- list(
    n = sum(!is.na(y)),
    sum = sum(y, na.rm = TRUE),
    sumsq = sum(y^2, na.rm = TRUE))
  class(ans) <- c("GaussianSuf", "Suf")
  return(ans)
}

RegressionSuf <- function(X = NULL,
                          y = NULL,
                          xtx = crossprod(X),
                          xty = crossprod(X, y),
                          yty = sum(y^2),
                          n = length(y),
                          xbar = colMeans(X),
                          ybar = mean(y)) {
  ## Sufficient statistics for a regression model.
  ## Args:
  ##   X:  The predictor matrix for a regression problem.
  ##   y:  The response vector for a regression problem.
  ##   xtx:  Cross product of the predictor matrix.
  ##   xty:  cross product of the predictor matrix with the response vector.
  ##   yty:  sum of squares of the response vector.
  ##   n:  sample size.
  ##   xbar:  column means of the predictor matrix.
  ##
  ## Returns:
  ##   A list containing the arguments, which are checked to ensure they have
  ##   legal values and sizes.
  if (!is.null(X)) {
    stopifnot(is.matrix(X))

    if (!is.null(y)) {
      stopifnot(is.numeric(y),
                length(y) == nrow(X))
    }
  }

  stopifnot(is.matrix(xtx),
            all(xtx == t(xtx)))
  stopifnot(is.numeric(xty),
            length(xty) == nrow(xtx))
  stopifnot(is.numeric(yty),
            length(yty) == 1,
            yty >= 0)
  stopifnot(is.numeric(n),
            length(n) == 1,
            n >= 0)
  stopifnot(is.numeric(xbar),
            length(xbar) == length(xty))

  ans <- list(xtx = xtx,
              xty = xty,
              yty = yty,
              n = n,
              xbar = xbar)
  class(ans) <- c("RegressionSuf", "Suf")
  return(ans)
}
