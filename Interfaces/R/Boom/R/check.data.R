## A collection of utilities for checking whether objects match
## various concepts.

check.scalar.probability <- function(x) {
  okay <- is.numeric(x) && length(x) == 1 && x >= 0 && x <= 1
  if (!okay) {
    stop("Expected a scalar probability.")
  }
  return(TRUE)
}

check.positive.scalar <- function(x) {
  okay <- is.numeric(x) && length(x) == 1 && x > 0
  if (!okay) {
    stop("Expected a positive scalar.")
  }
  return(TRUE)
}

check.nonnegative.scalar <- function(x) {
  okay <- is.numeric(x) && length(x) == 1 && x >= 0
  if (!okay) {
    stop("Expected a non-negative scalar.")
  }
  return(TRUE)
}

check.scalar.integer <- function(x) {
  okay <- is.numeric(x) && (length(x) == 1) && (abs(x - as.integer(x)) < 1e-10)
  if (!okay) {
    stop("Expected a scalar integer.")
  }
  return(TRUE)
}

check.scalar.boolean <- function(x) {
  if (!is.logical(x) || !(length(x) == 1)) {
    stop("Expected a scalar boolean value.")
  }
  return(TRUE)
}

check.probability.distribution <- function(x) {
  okay <- is.numeric(x) && isTRUE(all.equal(sum(x), 1)) && all(x >= 0)
  if (!okay) {
    stop("Expected a discrete probability distribution.")
  }
  return(TRUE)
}

