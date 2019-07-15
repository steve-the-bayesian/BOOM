BsplineBasis <- function(x, knots) {
  ## Args:
  ##   x:  A nummeric vector to be expanded.
  ##   knots:  A vector of knots.
  stopifnot(is.numeric(x), !any(is.na(x)))
  stopifnot(is.numeric(knots), !any(is.na(x)))
  knots <- sort(knots)
  ans <- .Call("boom_spike_slab_Bspline_basis",
               x,
               knots)
  attr(ans, "knots") <- knots
  class(ans) <- c("BsplineBasis", "SplineBasis")
  return(ans)
}

MsplineBasis <- function(x, knots) {
  ## Args:
  ##   x:  A nummeric vector to be expanded.
  ##   knots:  A vector of knots.
  stopifnot(is.numeric(x), !any(is.na(x)))
  stopifnot(is.numeric(knots), !any(is.na(x)))
  knots <- sort(knots)
  ans <- .Call(boom_spike_slab_Mspline_basis,
               x,
               knots)
  attr(ans, "knots") <- knots
  class(ans) <- c("MsplineBasis", "SplineBasis")
  return(ans)
}

IsplineBasis <- function(x, knots) {
  ## Args:
  ##   x:  A nummeric vector to be expanded.
  ##   knots:  A vector of knots.
  stopifnot(is.numeric(x), !any(is.na(x)))
  stopifnot(is.numeric(knots), !any(is.na(x)))
  knots <- sort(knots)
  ans <- .Call("boom_spike_slab_Ispline_basis",
               x,
               knots)
  attr(ans, "knots") <- knots
  class(ans) <- c("IsplineBasis", "SplineBasis")
  return(ans)
}

knots.SplineBasis <- function(Fn, ...) {
  ## Args:
  ##   Fn: A BOOM spline basis matrix.
  ##   ...:  Unused.  Required to match the signature of the 'knots' generic function.
  ##
  ## Returns:
  ##   The 'knots' attribute of Fn.
  return(attr(Fn, "knots"))
}
