# Copyright 2019 Steven L. Scott.  All rights reserved.
#
# Copyright 2018 Google LLC. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

TimestampInfo <- function(response, data = NULL, timestamps = NULL) {
  ## Args:
  ##   response: A vector or matrix.  If the response is a zoo object with
  ##     timestamps then the timestamps will be processed.
  ##   data: an optional data frame, list or environment (or object coercible by
  ##     ‘as.data.frame’ to a data frame) containing the variables in the bsts
  ##     model.  This is only used as source of timestamps if they are not
  ##     provided otherwise.
  ##   timestamps: A vector of user-supplied timestamps (of the same length as
  ##     'response'), or NULL.
  ##
  ## Returns:
  ##   An object of class TimestampInfo, which is a list containing the
  ##   following elements:
  ##
  ##   * timestamps.are.trivial: Logical.  TRUE indicates that 'timestamps' are
  ##     either NULL, or that there are no duplicates and no skipped values.  If
  ##     timestamps are trivial then the response is a simple uninterrupted
  ##     sequence (like you'd expect at time series to be.
  ##
  ##   * number.of.time.points: The number of unique time points contained in
  ##       the data, including any skipped time points containing no
  ##       observations.
  ## 
  ##   * timestamps: the time stamps taken from the original response or zoo
  ##       data frame containing the data.  This might be NULL.  It might also
  ##       contain duplicate values.
  ##
  ##   * regular.timestamps: A regular grid of timestamps containing the
  ##       smallest and largest and largest entries in 'timestamps', with an
  ##       increment equal to the smallest nonzero increment in 'timestamps'.
  ##       This will be NULL if 'timestamps' is NULL.
  ##
  ##   * timestamp.mapping: This is only present if timestamps.are.trivial is
  ##       FALSE.  It is a numeric vector giving the index of the entry in
  ##       'regular.timestamps' to which each observation in 'response' is
  ##       associated.
  if (is.null(timestamps)) {
    if (is.zoo(response)) {
      timestamps <- index(response)
    } else if (!is.null(data) && is.zoo(data)) {
      timestamps <- index(data)
    }
  }
  number.of.observations <-
      if (is.matrix(response)) nrow(response) else length(response)

  if (is.null(timestamps)) {
    ## Handle the trivial case when no timestamps are passed.
    number.of.time.points <- number.of.observations
    stopifnot(number.of.time.points > 0)
    ans <- list(timestamps.are.trivial = TRUE,
                number.of.time.points = number.of.time.points,
                timestamps = 1:number.of.observations,
                regular.timestamps = 1:number.of.time.points)
  } else {
    ## If timestamps were passed then process them.
    stopifnot(number.of.observations == length(timestamps))
    regular.timestamps <- RegularizeTimestamps(timestamps)
    ans <- list(timestamps.are.trivial = IsRegular(timestamps),
      number.of.time.points = length(regular.timestamps),
      timestamps = timestamps,
      regular.timestamps = regular.timestamps)
    if (!ans$timestamps.are.trivial) {
      ## A hack to handle numeric timestamps appropriately.
      class(timestamps) <- class(regular.timestamps)
      ans$timestamp.mapping <- zoo::MATCH(timestamps, regular.timestamps)
    }
    if (length(ans$regular.timestamps) > 2 * length(ans$timestamps)) {
      warning("Expanding the time series to a regular interval resulted ",
        "in very large amounts of missing data.")
    }
  }
  class(ans) <- "TimestampInfo"
  return(ans)
}

NoDuplicates <- function(timestamps) {
  ## Returns TRUE iff the vector of timestamps contains no duplicate values.
  return(length(timestamps) == length(unique(timestamps)))
}

HasDuplicateTimestamps <- function(bsts.object) {
  # Returns TRUE iff the object has nontrivial timestamps and at least one time
  # stamp is associated with more than one observation.
  if (bsts.object$timestamp.info$timestamps.are.trivial) {
    return(FALSE)
  }
  return(!NoDuplicates(bsts.object$timestamps))
}

NoGaps <- function(timestamps) {
  ## Returns TRUE iff there are no gaps in the series of time stamps.  The
  ## deltas between time points do not need to be equally spaced.  A distance
  ## between observations counts as a 'gap' if it is at least twice as large as
  ## the smallest duration between two time points.  If we take "twice"
  ## literally then we are vulnerable to floating point junk, so take a factor
  ## of 1.8 instead of 2.
  unique.timestamps <- unique(sort(timestamps))
  dt <- diff(unique.timestamps)
  min.dt <- min(dt)
  return(all(dt < 1.8 * min.dt))
}

IsRegular <- function(timestamps) {
  ## A sequence of time stamps is regular if it contains no duplicates and no
  ## gaps.
  return(NoDuplicates(timestamps) && NoGaps(timestamps))
}

RegularizeTimestamps <- function(timestamps) {
  ## Args:
  ##   timestamps: A set of timestamps.  This can be NULL, a numeric object
  ##     (presumably time 1, 2, 3), a numeric set of timestamps that came from a
  ##     ts object (like 1945.08333, 1945.127, ...), a Date object, or a POSIXt
  ##     object.
  ##
  ## Returns:
  ##   A vector of unique, regularly spaced time stamps containing the
  ##   (deduplicated) entries in the argument, and maybe more, if there were
  ##   some time steps passed over by the supplied time stamps.
  ##
  ##   Note that 'regularly spaced' time steps do not always have the same time
  ##   gap between them.  Monthly data is the most obvious example, but daily
  ##   POSIXt data passing over a DST boundary is another.
  if (is.null(timestamps) || IsRegular(timestamps)) {
    return(timestamps)
  }
  UseMethod("RegularizeTimestamps")
}

RegularizeTimestamps.default <- function(timestamps) {
  ## Args:
  ##   timestamps: A set of timestamps.  This can be NULL, a numeric object
  ##     (presumably time 1, 2, 3), a Date object, or a POSIXt object.
  ##
  ## Returns:
  ##   A vector of unique, regularly spaced time stamps containing the
  ##   (deduplicated) entries in the argument, and maybe more, if there were
  ##   some time steps passed over by the supplied time stamps.
  unique.timestamps <- sort(unique(timestamps))
  dt <- diff(unique.timestamps)
  dt <- min(dt)
  regular.timestamps <- seq(from = unique.timestamps[1],
                            to = tail(unique.timestamps, 1),
                            by = dt)
  if (!all(timestamps %in% regular.timestamps)) {
    stop("Time stamps must occur at regular intervals.")
  }
  return(regular.timestamps)
}

RegularizeTimestamps.numeric <- function(timestamps) {
  ## If the timestamps are numeric then we need to guard against floating point
  ## garbage.
  unique.timestamps <- sort(unique(timestamps))
  dt <- min(diff(unique.timestamps))
  regular.timestamps <- seq(from = unique.timestamps[1],
                            to = tail(unique.timestamps, 1),
                            by = dt)
  if (!all(signif(timestamps, digits = 8)  %in%
           signif(regular.timestamps, digits = 8))) {
    stop("Time stamps must occur at regular intervals.")
  }
  class(regular.timestamps) <- "NumericTimestamps"
  return(regular.timestamps)
}

MATCH.NumericTimestamps <- function(x, table, nomatch = NA, ...) {
  ## An S3 generic for the MATCH function provided by the zoo package.  Numeric
  ## timestamps match if they agree on 8 significant digits.
  match(signif(x, digits = 8),
        signif(table, digits = 8),
        nomatch = nomatch,
        ...)
}

RegularizeTimestamps.Date <- function(timestamps) {
  unique.timestamps <- sort(unique(timestamps))
  dt <- diff(unique.timestamps)
  min.dt <- min(dt)
  dt.table <- table(dt)
  if (min.dt == 1 || min.dt == 7) {
    ## Daily or weekly data is regular.
    return(RegularizeTimestamps.default(timestamps))
  } else if (min.dt %in% 28:31 &&
             any(dt.table[c("30", "31")] > 0)) {
    ## Monthly data
    regular.timestamps <- seq(from = unique.timestamps[1],
                              to = tail(unique.timestamps, 1),
                              by = "month")
  } else if (min.dt %in% 89:93) {
    regular.timestamps <- seq(from = unique.timestamps[1],
                              to = tail(unique.timestamps, 1),
                              by = "quarter")
  } else if (min.dt == 365) {
    regular.timestamps <- seq(from = unique.timestamps[1],
                              to = tail(unique.timestamps, 1),
                              by = "year")
  } else {
    stop("Could not determine an appropriate time scale.")
  }
  if (!all(timestamps %in% regular.timestamps)) {
    stop("Date time stamps must occur at regular intervals.")
  }
  return(regular.timestamps)
}

RegularizeTimestamps.POSIXt <- function(timestamps) {
  dt <- diff(sort(unique(timestamps)))
  if (attributes(dt)$units == "days") {
    days <- as.POSIXct(RegularizeTimestamps(as.Date(timestamps)))
    offset <- min(timestamps) - min(days)
    times <- as.POSIXlt(days + offset)
    dst <- as.logical(times$isdst)
    if (dst[1]) {
      ## If the first time period was in daylight savings time then all the time
      ## points not in daylight savings time need to be shifted back by an hour.
      times[!dst] <- times[!dst] + as.difftime(1, units = "hours")
    } else {
      ## If the first time period was in standard time then all the time
      ## points in daylight savings time need to be shifted forward by an hour.
      times[dst] <- times[dst] - as.difftime(1, units = "hours")
    }
    return(as.POSIXct(times))
  } else if(attributes(dt)$units == "hours") {
    ## If the difference in times is on the hourly scale, it might be because
    ## we've got daily data with DST effects yielding differences of 23, 24, and
    ## 25 hours.
    dt.table <- table(dt)
    min.dt <- min(dt)
    mode.dt <- as.numeric(names(dt.table))[which.max(dt.table)]
    if (min.dt == 23 && mode.dt == 24) {
      return(seq(from = min(timestamps), to = max(timestamps), by = "DSTday"))
    } else {
      return(RegularizeTimestamps.default(timestamps))
    }
  } else {
    return(RegularizeTimestamps.default(timestamps))
  }
}
