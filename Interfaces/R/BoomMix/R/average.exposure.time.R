# Copyright 2012 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

AverageExposureTime <- function(point.process.list,
                                timescale = c("days", "secs", "mins", "hours",
                                  "weeks"),
                                reduce = TRUE) {
  ## Computes the average amount of time each PointProcess in
  ## point.process.list has been observed.
  ## Args:
  ##   point.process.list: Either a single PointProcess object, or a list
  ##     of such objects
  ##   timescale: The time scale to use when computing time differences.
  ##   reduce: If TRUE then a grand mean will be computed from the
  ##     list of inputs.  If FALSE then a vector is returned with the
  ##     exposure time for each element in the point.process.list.
  ## Returns:
  ##   See the comments under the 'reduce' argument.
  if (inherits(point.process.list, "PointProcess")) {
    point.process.list <- list(point.process.list)
  }
  stopifnot(is.list(point.process.list))
  stopifnot(all(sapply(point.process.list, inherits, "PointProcess")))
  timescale <- match.arg(timescale)
  .GetExposureTime <- function(x) {
    return(difftime(x$end, x$start, units = timescale))
  }
  exposure.time <- sapply(point.process.list, .GetExposureTime)
  if (reduce) {
    return(mean(exposure.time))
  } else{
    return(exposure.time)
  }
}
