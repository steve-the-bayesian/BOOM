.MakeKnots <- function(x, numknots) {
  ## Return a vector of quantiles of x of length numknots, equally space in
  ## quantile space.
  stopifnot(is.numeric(numknots), length(numknots) == 1, numknots > 0)
  stopifnot(is.numeric(x), !any(is.na(x)))
  knot.quantiles <- seq(1/numknots, 1 - 1/numknots, length = numknots)
  knots <- quantile(x, knot.quantiles)
  return(knots)
}

BsplineBasis <- function(x, knots = NULL, numknots = 3) {
  ## Args:
  ##   x:  A nummeric vector to be expanded.
  ##   knots:  A vector of knots.
  ##   numknots:  If 'knots' is supplied this is ignored.
  stopifnot(is.numeric(x), !any(is.na(x)))

  if (is.null(knots)) {
    knots <- .MakeKnots(x, numknots)
  }
  stopifnot(is.numeric(knots))
  knots <- sort(knots)
  ans <- .Call("boom_spike_slab_Bspline_basis",
               x,
               knots)
  attr(ans, "knots") <- knots
  class(ans) <- c("BsplineBasis", "SplineBasis")
  return(ans)
}

MsplineBasis <- function(x, knots = NULL, numknots = 3) {
  ## Args:
  ##   x:  A nummeric vector to be expanded.
  ##   knots:  A vector of knots.
  stopifnot(is.numeric(x), !any(is.na(x)))
  if (is.null(knots)) {
    knots <- .MakeKnots(x, numknots)
  }
  stopifnot(is.numeric(knots))
  knots <- sort(knots)
  ans <- .Call(boom_spike_slab_Mspline_basis,
               x,
               knots)
  attr(ans, "knots") <- knots
  class(ans) <- c("MsplineBasis", "SplineBasis")
  return(ans)
}

IsplineBasis <- function(x, knots = NULL, numknots = 3) {
  ## Args:
  ##   x:  A nummeric vector to be expanded.
  ##   knots:  A vector of knots.
  stopifnot(is.numeric(x), !any(is.na(x)))
  if (is.null(knots)) {
    knots <- .MakeKnots(x, numknots)
  }
  stopifnot(is.numeric(knots))
  knots <- sort(knots)
  ans <- .Call("boom_spike_slab_Ispline_basis",
               x,
               knots)
  attr(ans, "knots") <- knots
  class(ans) <- c("IsplineBasis", "SplineBasis")
  return(ans)
}

knots <- function(Fn, ...) {
  UseMethod("knots")
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
