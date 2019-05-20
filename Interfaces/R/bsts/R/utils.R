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

.NonzeroCols <- function(mat) {
  ## Return the columns of mat that are not identically zero.
  all.zero <- apply(mat == 0, 2, all)
  return(mat[, !all.zero])
}

.FindStateSpecification <- function(state.specification, bsts.object) {
  ## Return the position of spec in bsts.object$state.specification.  If spec
  ## does not exist return NULL.
  stopifnot(inherits(bsts.object, "bsts"))
  stopifnot(inherits(state.specification, "StateModel"))
  IsEqual <- function(thing1, thing2) {
    ## A helper function that can compare two lists for equality.
    eq <- all.equal(thing1, thing2)
    return(is.logical(eq) && eq)
  }
  for (i in 1:length(bsts.object$state.specification)) {
    if (IsEqual(state.specification, bsts.object$state.specification[[i]])) {
      return(i);
    }
  }
  return(NULL);
}

.ScreenMatrix <- function(nrow, ncol, side.margin = .07, top.margin = .07) {
  ## Creates a matrix that can be passed to split.screen, with a rectangular
  ## array of screens similar to that created by mfrow(nrow, ncol).
  ##
  ## Args:
  ##   nrow:  The desired number of rows of screens.
  ##   ncol:  The desired number of columns of screens.
  ##   side.margin: The fraction of the display devoted to the side margin.
  ##     This margin will be present on both sides.
  ##   top.margin: The fraction of the display devoted to the top margin.  An
  ##     equivalent amount of space will be given to the bottom margin.
  ##
  ## Each row in the screen.matrix defines the left, right, bottom, and top
  ## coordinates of a screen.
  plot.width <- (1 - 2 * side.margin) / ncol
  top.margin <- .07
  plot.height <- (1 - 2 * top.margin) / nrow
  counter <- 1
  screens <- list()
  bottom.coordinate <- 1 - top.margin - plot.height
  for (i in 1:nrow) {
    left.coordinate <- side.margin
    for (j in 1:ncol) {
      screens[[counter]] <-
        c(left.coordinate,
          left.coordinate + plot.width,
          bottom.coordinate,
          bottom.coordinate + plot.height)
      left.coordinate <- left.coordinate + plot.width
      counter <- counter + 1
    }
    bottom.coordinate <- bottom.coordinate - plot.height
  }
  screen.matrix <- matrix(nrow = nrow * ncol, ncol = 4)
  for (i in 1:nrow(screen.matrix)) {
    screen.matrix[i, ] <- screens[[i]]
  }
  return(screen.matrix)
}

##===========================================================================
.AddDateAxis <- function(time, position = 1) {
  ## Add a date axis to a plot.
  ## Args:
  ##   time: The times to plot on the axis tickmarks.  Can be Date, POSIXt, or
  ##     numeric.
  ##   position: The axis position 1 - 4, corresponding to the x, y, top, and
  ##     right axes.
  ## Effects:
  ##   An axis is added to the current plot.
  if (inherits(time, "Date")) {
    axis.Date(position, time, xpd = NA)
  } else if (inherits(time, "POSIXt")) {
    axis.POSIXct(position, as.POSIXct(time), xpd = NA)
  } else {
    axis(position, xpd = NA)
  }
  return(invisible(NULL))
}

as.Date.POSIXct <- function(x, ...) {
  ## Convert a POSIXct object to a Date, without shifting time zones.
  return(base::as.Date.POSIXct(x, tz = Sys.timezone()))
}
as.Date.POSIXlt <- function(x, ...) {
  ## Convert a POSIXlt object to a Date, without shifting time zones.
  return(base::as.Date.POSIXlt(x, tz = Sys.timezone()))
}

YearMonToPOSIX <- function(timestamps) {
  ## Convert an object of class yearmon to class POSIXt, without getting bogged
  ## down in timezone calculations.
  ##
  ## Calling as.POSIXct on another date/time object (e.g. Date) applies a
  ## timezone correction to the object.  This can shift the time marker by a few
  ## hours, which can have the effect of shifting the day by one unit.  If the
  ## day was the first or last in a month or year, then the month or year will
  ## be off by one as well.
  ##
  ## Coercing the object to the character representation of a Date prevents this
  ## adjustment from being applied, and leaves the POSIXt return value with the
  ## intended day, month, and year.
  stopifnot(inherits(timestamps, "yearmon"))
  return(as.POSIXct(as.character(as.Date(timestamps))))
}

DateToPOSIX <- function(timestamps) {
  ## Convert an object of class Date to class POSIXct without getting bogged
  ## down in timezone calculation.
  ##
  ## Calling as.POSIXct on another date/time object (e.g. Date) applies a
  ## timezone correction to the object.  This can shift the time marker by a few
  ## hours, which can have the effect of shifting the day by one unit.  If the
  ## day was the first or last in a month or year, then the month or year will
  ## be off by one as well.
  ##
  ## Coercing the object to the character representation of a Date prevents this
  ## adjustment from being applied, and leaves the POSIXt return value with the
  ## intended day, month, and year.
  stopifnot(inherits(timestamps, "Date"))
  return(as.POSIXct(as.character(timestamps)))
}

###----------------------------------------------------------------------
StateSizes <- function(state.specification) {
  ## Returns a vector giving the number of dimensions used by each state
  ## component in the state vector.
  ## Args:
  ##   state.specification: a vector of state specification elements.
  ##     This most likely comes from the state.specification element
  ##     of a bsts.object
  ## Returns:
  ##   A numeric vector giving the dimension of each state component.
  state.component.names <- sapply(state.specification, function(x) x$name)
  state.sizes <- sapply(state.specification, function(x) x$size)
  if (any(is.na(state.sizes)) ||
        any(is.null(state.sizes)) ||
        any(!is.numeric(state.sizes))) {
    stop("One or more state components were missing the 'size' attribute.")
  }
  names(state.sizes) <- state.component.names
  return(state.sizes)
}

###----------------------------------------------------------------------
SuggestBurn <- function(proportion, bsts.object) {
  ## Suggests a size of a burn-in sample to be discarded from the MCMC
  ## run.
  ## Args:
  ##   proportion: A number between 0 and 1.  The fraction of the run
  ##     to discard.
  ##   bsts.object:  An object of class 'bsts'.
  ## Returns:
  ##   The number of iterations to discard.
  niter <- bsts.object$niter
  if (is.null(niter)) {
    ## Modern bsts objects will all have a 'niter' member.  Serialized
    ## bsts objects that were fit long ago expect niter to be the
    ## length of sigma.obs.
    niter <- length(bsts.object$sigma.obs)
  }
  if (is.null(bsts.object$log.likelihood)) {
    burn <- floor(proportion * niter)
  } else {
    burn <- SuggestBurnLogLikelihood(bsts.object$log.likelihood, proportion)
  }
  if (burn >= niter) {
    warning(paste0("SuggestBurn wants to discard everything\nn = ", niter,
                   "proportion = ", proportion, "."))
    if (niter > 0) {
      ## You have to keep at least one observation.
      burn <- niter - 1
    } else {
      ## You have to burn at least one observation.
      burn <- 1
    }
  }
  if (burn == 0) {
    ## This is to keep Kay's tests happy.
    burn <- 1
  }
  return(burn)
}
###----------------------------------------------------------------------
Shorten <- function(words) {
  ## Removes a prefix and suffix common to all elements of 'words'.
  ## Args:
  ##   words:  A character vector.
  ##
  ## Returns:
  ##   'words' with common prefixes and suffixes removed.
  ##
  ## Details:
  ##   The typical use case for this function is factor level names
  ##   that are (e.g.) names files from the same directory with
  ##   similar suffixes.  The common prefix (file path) gets removed,
  ##   as does the common suffix (.tex).
  ##
  ##   Shorten(c("/usr/common/foo.tex", "/usr/common/barbarian.tex")
  ##   produces c("foo", "barbarian")
  if (is.null(words)) return (NULL)
  stopifnot(is.character(words))
  if (length(unique(words)) == 1) {
    ## If all the words are the same don't do any shortening.
    return(words)
  }

  first.letters <- substring(words, 1, 1)
  while (all(first.letters == first.letters[1])) {
    words <- substring(words, 2)
    first.letters <- substring(words, 1, 1)
  }

  word.length <- nchar(words)
  last.letters <- substring(words, word.length, word.length)
  while (all(last.letters == last.letters[1])) {
    words <- substring(words, 1, word.length - 1)
    word.length <- word.length - 1
    last.letters <- substring(words, word.length, word.length)
  }

  return(words)
}

.SetTimeZero <- function(time0, y) {
  ## Args:
  ##   time0:  A timestamp to use as the answer, or NULL.
  ##   y: The time series being modeled.  If time0 is NULL and y is of type zoo
  ##     then the index of y[1] will be used as time0.
  ##
  ## Returns:
  ##   A timestamp of class POSIXt.
  if (is.null(time0)) {
    if (is.null(y)) {
      stop("You must supply time0 if y is missing.")
    }
    if (!inherits(y, "zoo")) {
      ## Note:  an xts object inherits from zoo.
      stop("You must supply 'time0' if y is not a zoo or xts object.")
    }
    time0 <- index(as.xts(y))[1]
  }

  if (inherits(time0, "Date")) {
      ## Converting dates to characters keeps as.POSIXct from changing the date
      ## in time zones with negative offsets.
    time0 <- as.character(time0)
  }

  tryCatch(time0 <- as.POSIXct(time0),
    error = simpleError(
      "The supplied or inferred time0 could not be converted to POSIXct."))
  stopifnot(inherits(time0, "POSIXt"))
  return(time0)
}

LongToWideArray <- function(predictor.matrix, series.id, timestamps) {
  ## Convert a matrix of predictors into an array with dimensions [series, time,
  ## xdim]
  ##
  ## Args:
  ##   predictor.matrix:  A matrix of predictor variables.
  ##   series.id: A factor indicating the series to which each row of
  ##     predictor.matrix belongs.
  ##   timestamps: A vector of timestamps indicating the time period to which
  ##     each row of predictor.matrix belongs.
  ##
  ## Returns:
  ##   A 3-way array containing the information in predictor.matrix, organized
  ##   according to series and time.
  unique.times <- sort(unique(timestamps))
  series.id <- as.factor(series.id)
  unique.names <- levels(series.id)
  ntimes <- length(unique.times)
  nseries <- length(unique.names)
  xdim <- ncol(predictor.matrix)
  
  ans <- array(NA, dim = c(nseries, ntimes, xdim))
  dimnames(ans) <- list(unique.names, as.character(unique.times),
    colnames(predictor.matrix))
  for (series in 1:nseries) {
    index <- as.numeric(series.id) == series
    series.predictors <- predictor.matrix[index, , drop = FALSE]
    series.timestamps <- as.character(timestamps[index])
    ans[series, series.timestamps, ] <- series.predictors
  }
  return(ans)
}

LongToWide <- function(response, series.id, timestamps) {
  ## Convert a multivariate time series in "long" format to "wide" format.
  ##
  ## Args:
  ##   response:  The time series values.
  ##   series.id: A vector of labels of the same length as 'response' indicating
  ##     the time series to wihch each element of 'response' belongs.
  ##   timestamps:  The time period to which each observation belongs.
  ##
  ## Returns:
  ##   A zoo matrix with rows corresponding to time stamps and columns
  ##   corresponding to different time series.  The matrix elements are the
  ##   'response' values.
  ##
  ## Note:
  ##   This could be done with 'reshape'.  I have reworked things by hand in the
  ##   interest of readability.
  stopifnot(length(response) == length(series.id),
    length(response) == length(timestamps))
  unique.times <- sort(unique(timestamps))
  series.id <- as.factor(series.id)
  unique.names <- levels(series.id)
  ntimes <- length(unique.times)
  nseries <- length(unique.names)
  
  ans <- matrix(nrow = ntimes, ncol = nseries)
  if (ntimes == 0 || nseries == 0) {
    return(ans)
  }
  colnames(ans) <- as.character(levels(series.id))

  for (i in 1:ntimes) {
    index <- timestamps == unique.times[i]
    observed <- as.character(series.id[index])
    ans[i, observed] <- response[index]
  }
  ans <- zoo(ans, unique.times)
  return(ans)
}

WideToLong <- function(response, na.rm = TRUE) {
  ## Convert a multiple time series in wide format, to long format.
  ##
  ## Args:
  ##   respponse: A time series matrix (or zoo matrix).  Rows represent time
  ##     points.  Columns are different series.
  ##   na.rm: If TRUE then missing values in the time series matrix will be
  ##     omitted from the returned data frame.
  ##
  ## Returns:
  ##   A data frame in "long" format containing three columns:
  ##   - The first column contains the timestamps.
  ##   - The second column contains a factor indicating which column is being
  ##      measured.
  ##   - The third column contains the value of the time series.
  ##
  ## Note:
  ##   This could be done with 'reshape'.  I have reworked things by hand in the
  ##   interest of readability.
  stopifnot(is.matrix(response))
  if (nrow(response) == 0) {
    return(NULL)
  }
  nseries <- ncol(response)
  
  if (is.zoo(response)) {
    timestamps <- index(response)
  } else {
    timestamps <- 1:nrow(response);
  }
  vnames <- colnames(response)
  if (is.null(vnames)) {
    vnames <- base::make.names(1:nseries)
  }
  ntimes <- length(unique(timestamps))
  
  values <- as.numeric(t(response))
  labels <- factor(rep(vnames, times = ntimes), levels = vnames)
  timestamps <- rep(timestamps, each = nseries)
  
  ans <- data.frame("time" = timestamps, "series" = labels, "values" = values)
  if (na.rm) {
    missing <- is.na(values)
    ans <- ans[!missing, ]
  }
  return(ans)
}

