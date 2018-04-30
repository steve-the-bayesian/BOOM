# Copyright 2012 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

AverageEventRate <- function(point.process,
                             timescale = c("days", "secs", "mins", "hours",
                               "weeks"),
                             reduce = TRUE) {
  ## Computes the average event rate from a the list of PointProcess objects.
  ## Args:
  ##   point.process: Either a single PointProcess object, or a list
  ##     of such objects
  ##   timescale: The scale of the denominator to use when computing
  ##     the average number of events per unit of time.
  ##   reduce: If TRUE then a grand mean will be computed from the
  ##     list of inputs.  If FALSE then a vector is returned with the
  ##     average event rate for each element in the point.process
  ##     list.  If a single PointProcess was supplied then this
  ##     argument is irrelevant.
  ## Returns:
  ##   A vector giving the average number of events per unit of time.
  stopifnot(is.list(point.process))
  if (inherits(point.process, "PointProcess")) {
    ## Allow single PointProcess objects to be passed
    point.process <- list(point.process)
  }

  timescale <- match.arg(timescale)
  stopifnot(length(point.process) > 0)
  stopifnot(all(sapply(point.process, inherits, "PointProcess")))
  .GetExposureTime <- function(x) {
    return(as.numeric(difftime(x$end, x$start, units = timescale)))}
  exposure.time <- sapply(point.process, .GetExposureTime)
  number.of.events <- sapply(point.process, function(x) length(x$events))
  if (reduce) {
    return(sum(number.of.events) / sum(exposure.time))
  } else {
    return(number.of.events / exposure.time)
  }
}
