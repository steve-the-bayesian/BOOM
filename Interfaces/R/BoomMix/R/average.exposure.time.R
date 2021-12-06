# Copyright 2021 Steven L. Scott. All Rights Reserved.
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
