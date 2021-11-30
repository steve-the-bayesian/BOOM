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


PointProcess <- function(events, start = NULL, end = NULL, group.id = NULL) {
  ## Constructor for PointProcess objects.
  ##
  ## A PointProcess is an ordered series of event times, paired with a
  ## beginning and ending time for an observation window.
  ##
  ## Args:
  ##   events: A vector of event times convertible to POSIXct.  The
  ##     times must be sorted from earliest to latest.
  ##   start: An optional time (convertible to POSIXct) for the start
  ##     of the observation window.  If missing, 'start' is taken to
  ##     be the first element of 'events'
  ##   end: An optional time (convertible to POSIXct) for the end of
  ##     the observation window.
  ##   group.id: An optional factor that can be used to create a list
  ##     of point process objects (one for each distinct level of
  ##     'group.id', each with the same 'start' and 'end'.
  ## Returns:
  ##   If group.id is NULL, then this returns an object of class
  ##   PointProcess, which is a list with elements events, start, and
  ##   end.  If group.id is non-null then this function returns a list
  ##   of PointProcess.
  if (!is.null(group.id)) {
    events <- split(events, group.id)
    ans <- lapply(events, PointProcess,
                  start = start, end = NULL, group.id = NULL)
    return(ans)
  }

  timestamps <- as.POSIXct(events)
  if (is.unsorted(timestamps)) {
    stop("events must be sorted")
  }

  if (is.null(start)) start <- timestamps[1]
  if (is.null(end)) end <- tail(timestamps, 1)
  start <- as.POSIXct(start)
  end <- as.POSIXct(end)
  stopifnot(start <= timestamps[1])
  stopifnot(end >= tail(timestamps, 1))
  ans <- list(events = timestamps, start = start, end = end);
  class(ans) <- "PointProcess"
  return(ans)
}

length.PointProcess <- function(x) {
  ## S3 generic defining the length of a PointProcess as the number of
  ## events.
  return(length(x$events))
}

density.PointProcess <- function(x, from = x$start, to = x$end, ...) {
  ## S3 generic for computing the density of a point process.
  ## Args:
  ##   x:  An object of class PointProcess.
  ##   from: A time point (convertible to POSIXct) for the beginning
  ##     of the density estimate.
  ##   to: A time point (convertible to POSIXct) for the end of the
  ##     density estimate.
  ##   ...: additional arguments passed to density().
  ## Returns:
  ##   An object of class PointProcessDensity, which is a density
  ##   object where the area under the curve is the number of points
  ##   in x.
  events <- x$events
  if (is.null(from)) from <- x$start

  origin <- x$start - as.numeric(x$start)
  if (is.null(to)) to <- x$end

  events <- events[events >= from & events <= to]
  d <- density(as.numeric(events), ...)
  d$x <- as.POSIXct(d$x, origin = origin)
  d$y <- d$y * length(events)
  class(d) <- c("PointProcessDensity", class(d))

  return(d)
}

plot.PointProcessDensity <- function(x, main = NULL, xlab = NULL,
                                      ylab = "Density", type = "l",
                                      zero.line = TRUE, ...) {
  ## S3 generic for plotting the density of a PointProcess.  This
  ## function is very similar to plot.density, but it makes sure the
  ## horizontal axis is plotted with time/date symbols instead of
  ## their internal numeric codes (i.e. seconds since Jan 1 1970).
  ## Args:
  ##   x:  A PointProcessDensity object.
  ##   main:  Main figure title.
  ##   xlab:  Label for horiontal axis.
  ##   ylab:  Label for vertical axis.
  ##   type:  Type of figure to draw.
  ##   zero.line:  Logical.  Should a horizontal line be drawn at zero?
  ##   ...:  Extra arguments passed to plot().
  ## Returns:
  ##   NULL.  Draws a figure on the current graphics device.
  if (is.null(xlab))
    xlab <- paste("N =", x$n)
  if (is.null(main))
    main <- "Intensity of Events"
  plot(x$x, x$y, main = main, xlab = xlab, ylab = ylab, type = type,
       ...)
  if (zero.line)
    abline(h = 0, lwd = 0.1, col = "gray")
  invisible(NULL)
}

plot.PointProcess <- function(x, y = c("density", "counting", "jitter"),
                              style = y,
                              start = x$start, end = x$end,
                              kernel.scale = .1, ...) {
  ## S3 generic for plotting a PointProcess.
  ## Args:
  ##   x:  The object of class PointProcess to plot.
  ##   y:  A character string indicating the style of the plot.
  ##   style:  A synonym for y.  If both are given, y is ignored.
  ##   start:  Start time for the plot.
  ##   end:  End time for the plot.
  ##   kernel.scale: A factor by which to scale the default bandwidth used
  ##     by 'density' when style == "density".  Smaller values mean
  ##     greater resolution.  Larger values mean less smoothing.
  ##   ...:  Extra arguments passed to plot().
  ## Returns:
  ##   NULL.  Draws a figure on the current graphics device.
  stopifnot(inherits(x, "PointProcess"))
  times <- x$events
  style <- match.arg(style, c("density", "counting", "jitter"))
  if (style == "counting") {
    plot(times, 1:length(times), xlab = "Time", ylab = "Cum. Event Count",
         xlim = c(start, end), ...)
  } else if (style == "jitter") {
    y <- runif (length(times), 0, .1)
    plot(times, y, xlab = "Time", ylab = "Random Jitter",
         xlim = c(start, end), ylim = c(0, .2), ...)
  } else if (style == "density") {
    plot(density(x, from = start, to = end, adjust = kernel.scale), ...)
    rug(x$events)
  } else {
    stop("Unrecognized 'style' argument: ", style,
         " passed to plot.PointProcess")
  }
  return(invisible(NULL))
}

WeeklyProfile <- function(timestamps) {
  ## Produces an object summarizeing the weekly seasonal pattern in
  ## point process data.
  ## Args:
  ##   timestamps: This can either be a vector of time points coercible to
  ##     POSIXt, or an object of class PointProcess.
  ## Returns:
  ##   A WeeklyProfile object containing summaries of the weekly
  ##   seasonal pattern in timestamps.  The summaries include counts of the
  ##   number of times an event occurred on each day of the week and
  ##   on each hour of the day.  The hourly counts are split by
  ##   weekends and weekdays.
  if (inherits(timestamps, "PointProcess")) {
    timestamps <- timestamps$events
  }
  timestamps <- as.POSIXlt(sort(timestamps))
  dow <- weekdays(timestamps)
  day.names <- c("Sunday", "Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday")
  weekend <- dow %in% c("Saturday", "Sunday")
  dow <- table(factor(dow, levels = day.names))

  weekend.hours <- table(factor(timestamps[weekend]$hour, levels = 0:23))
  weekday.hours <- table(factor(timestamps[!weekend]$hour, levels = 0:23))

  ans <- list(times = timestamps,
              day.of.week = dow,
              weekday.hours = weekday.hours,
              weekend.hours = weekend.hours)
  class(ans) <- c("WeeklyProfile", class(ans))
  return(ans)
}

print.WeeklyProfile <- function(x, ...) {
  ## S3 generic for printing a WeeklyProfile object
  ## Args:
  ##   x:  An object of class WeeklyProfile to be printed
  ##   ...: Extra args are ignored.  This argument is present to match
  ##     the signature of the default print() method.
  ## Returns:
  ##   Prints x to the screen.  Invisibly returns x.
  print(x$day.of.week)
  m <- cbind(x$weekday.hours, x$weekend.hours)
  rownames(m) <- paste(0:23)
  colnames(m) <- c("weekday", "weekend")
  print(m)
  return(invisible(x))
}

plot.WeeklyProfile <- function(x, ...) {
  ## S3 generic for plotting a WeeklyProfile object.
  ## Args:
  ##   x:  An object of class WeeklyProfile to be plotted.
  ##   ...:  Extra arguments are ignored.
  ## Returns:
  ##   NULL.  Draws a figure on the current graphics device.
  profile <- x
  stopifnot(inherits(profile, "WeeklyProfile"))
  layout.matrix <- matrix(c(1, 2,   # counting process
                            3, 3),
                          byrow = TRUE, ncol = 2)
  layout(layout.matrix)
  on.exit(layout(matrix(1)))
  day.names <- c("Sunday", "Monday", "Tuesday", "Wednesday",
                 "Thursday", "Friday", "Saturday")
  plot(profile$times, 1:length(profile$times), pch = ".",
       xlab = "time", ylab = "Cumulative Events")
  barplot(profile$day.of.week, names.arg = substr(day.names, 1, 3),
          main = "Daily Pattern", las = 1)
  m <- cbind(profile$weekday.hours, profile$weekend.hours)
  barplot(t(m), besid = TRUE, , main = "Hourly Pattern",
          leg = c("Weekday", "Weekend"),
          args.legend = list(x = "topright", ncol = 2))
}
