Intensity <- function(x, ...) {
  ## Compute the intensity function for a vector of event times.
  ##
  ## Args:
  ##   x: An object representing a vector of time points.  This must be an
  ##     object of class 'POSIXt' or 'Date', both of which can be coerced to
  ##     numeric.
  ##   ...: Extra arguments passed to 'density'.
  ##
  ## Returns:
  ##   The intensity function describing the vector of timestamps.  This is a
  ##   dressed up 'density' object that has been renormalized so that the area
  ##   under the curve is 'n' (the number of events) instead of 1, and made
  ##   aware of the timestamps' original class.  The returned object has class
  ##   'Intensity', for which a plot method is provided.
  if (!inherits(x, "POSIXt") && !inherits(x, "Date")) {
    stop("Intensity only supports POSIXt and Date classes.")
  }
  ## Capture the name of 'x' before it is evaluated, for use as a default axis
  ## label.  'deparse' (rather than the R >= 4.0.0 'deparse1') is used so the
  ## package continues to build under the R version declared in DESCRIPTION.
  data.name <- paste(deparse(substitute(x)), collapse = " ")

  d <- density(as.numeric(x), na.rm = TRUE, ...)
  ## 'density' integrates to 1.  Rescale so the area under the curve is the
  ## number of events, turning the density estimate into an intensity estimate.
  d$y <- d$y * length(x)

  ## Restore the timestamps to their original class so that the plot method can
  ## draw a meaningful (time-aware) horizontal axis.
  if (inherits(x, "POSIXt")) {
    d$x <- as.POSIXct(d$x)
  } else {
    d$x <- as.Date(d$x)
  }

  d$data.name <- data.name
  class(d) <- "Intensity"
  return(d)
}


plot.Intensity <- function(x,
                           y = NULL,
                           xlab = NULL,
                           ylab = NULL,
                           type = "l",
                           zero.line = TRUE,
                           ...) {
  ## Plot method for objects of class 'Intensity'.
  ##
  ## Args:
  ##   x: An object of class 'Intensity', as produced by 'Intensity'.
  ##   y: Unused.  Present to match the signature of the generic 'plot'.
  ##   xlab: Character label for the horizontal axis.  If NULL, the name of the
  ##     variable passed to 'Intensity' is used.
  ##   ylab: Character label for the vertical axis.  If NULL, "Intensity" is
  ##     used.
  ##   type: The type of plot to draw.  See 'plot.default'.
  ##   zero.line: Logical.  If TRUE a dotted horizontal line is drawn at zero.
  ##   ...: Extra arguments passed to 'plot'.
  ##
  ## Returns:
  ##   Called for its side effect, which is to produce a plot.
  if (is.null(ylab)) {
    ylab <- "Intensity"
  }

  if (is.null(xlab)) {
    xlab <- x$data.name
  }

  plot(x$x, x$y, xlab = xlab, ylab = ylab, type = type, ...)
  if (zero.line) {
    abline(h = 0, lty = 3)
  }
}
