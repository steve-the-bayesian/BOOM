circles <- function(center, radius, ...) {
  ## Args:
  ##   center: A two-column matrix giving the coordinates of the circle center.
  ##     If a single circle is to be drawn then a 2-element vector can be passed
  ##     instead.
  ##   radius:  The radii of the circles.
  ##   ...: Extra arguments passed to 'segments'.  See 'par' for options
  ##     controlling line type, line width, color, etc.
  ##
  ## Effects:
  ##   One or more circles are drawn on the current graphics device at the given
  ##   locations.
  ##
  ## Returns:
  ##   Invisibly returns NULL.
  if (!is.matrix(center)) {
    stopifnot(is.numeric(center), length(center == 2))
    center <- matrix(center, nrow = 1)
  }
  stopifnot(is.matrix(center), ncol(center) == 2)

  stopifnot(is.numeric(radius))
  if (nrow(center) == 1 && length(radius) > 1) {
    center <- matrix(rep(center, 2 * length(radius)),
      nrow = length(radius),
      byrow = TRUE)
  }
  if (nrow(center) > 1 && length(radius) == 1) {
    radius <- rep(radius, nrow(center))
  }
  stopifnot(length(radius) == nrow(center))
  
  theta <- seq(0, 2 * pi, length = 50)
  unit.circle <- cbind(cos(theta), sin(theta))

  for (i in 1:length(radius)) {
    circle <- radius[i] * unit.circle
    circle[, 1] <- circle[, 1] + center[i, 1]
    circle[, 2] <- circle[, 2] + center[i, 2]
    rotated.index <- c(nrow(circle), 1:(nrow(circle) - 1))
    segments(circle[, 1], circle[, 2],
      circle[rotated.index, 1], circle[rotated.index, 2], ...)
  }
  return(invisible(NULL))
}
