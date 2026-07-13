.CollapseTable <- function(counts, max.levels, alphabetical = TRUE) {
  ## Truncates a table to have at most max.levels of entries.  The
  ## other entries will be collapsed into a single level called
  ## "other".  If alphabetical == TRUE then the returned table will be
  ## sorted alphabetically according to the table name.  Otherwise it
  ## will be sorted by frequency, in decreasing order.
  if (length(counts) <= max.levels) {
    return(if (alphabetical) .OrderAlphabetically(counts)
           else .OrderByFrequency(counts))
  }
  index <- rev(order(counts))
  new.tab <- counts[index][1:(max.levels - 1)]
  other <- sum(counts[index][-(1:max.levels - 1)])
  if (alphabetical) {
    alpha.index <- order(names(new.tab))
    new.tab <- new.tab[alpha.index]
  }
  new.tab <- c(new.tab, other)

  ## Make sure the last entry has a name, and its name is 'other'
  names(new.tab)[length(new.tab)] <- "other"

  return(new.tab)
}

.OrderAlphabetically <- function(counts) {
  ## Orders the 'table' object counts alphabetically by name.
  index <- order(names(counts))
  return(counts[index])
}

.OrderByFrequency <- function(counts) {
  ## Puts the 'table' object counts in decreasing order by frequency.
  return(rev(sort(counts)))
}

histabunch <- function(x, gap = 1, same.scale = FALSE, boxes = FALSE,
                       min.continuous = 12,
                       max.factor = 40,
                       vertical.axes = FALSE, ...){
  ## Plot a bunch of histograms describing the data in x.
  ## Args:
  ##   x:  A matrix or data frame containing the variables to be plotted.
  ##   gap:  The gap between the plots, measured in lines of text.
  ##   same.scale: Logical value indicating whether the histograms
  ##     should all be plotted on the same scale.
  ##   boxes: Logical value indicating whether boxes should be drawn
  ##     around the histograms.
  ##   min.continuous: Numeric variables with more than min.continuous
  ##     unique values will be plotted as continuous.  Otherwise they
  ##     will be plotted as factors.
  ##   max.factor: Factors with more than 'max.factor' levels will be
  ##     beautified (ha!) by combining their remaining levels into an
  ##     "other" category.
  ##   vertical.axes: Logical value indicating whether the histograms
  ##     should be given vertical "Y" axes.
  ##   ...: extra arguments passed to hist (for numeric variables) or
  ##     barplot (for factors).
  stopifnot(is.data.frame(x) || is.matrix(x))
  number.of.variables <- ncol(x)
  number.of.rows <- max(1, floor(sqrt(number.of.variables)))
  number.of.columns <- ceiling(number.of.variables / number.of.rows)

  if (!is.null(names(x))) {
    vnames <- names(x)
  } else if (!is.null(dimnames(x)[[2]])) {
    vnames <- colnames(x)
  } else {
    vnames <- paste("V", 1:number.of.variables, sep="")
  }

  is.continuous <- function(x){
    if (is.factor(x)) return(FALSE)
    if (!is.numeric(x)) return(FALSE)
    if (length(unique(x)) < min.continuous) return(FALSE)
    return(TRUE)
  }

  as.datetime <- function(x) {
    ## Interpret x as a vector of Date or POSIXt timestamps, if possible.
    ##
    ## Args:
    ##   x: A variable from the data being plotted.
    ##
    ## Returns:
    ##   A vector of class 'Date' or 'POSIXt' when x already carries such a
    ##   class, or when x is a character or factor vector whose values parse
    ##   cleanly as date-times or dates.  Returns NULL when x cannot be
    ##   interpreted as a sequence of timestamps.
    if (inherits(x, "POSIXt") || inherits(x, "Date")) {
      return(x)
    }
    if (!is.character(x) && !is.factor(x)) {
      return(NULL)
    }
    vals <- as.character(x)
    present <- !is.na(vals) & trimws(vals) != ""
    if (!any(present)) {
      return(NULL)
    }
    ## Prefer a date-time interpretation, trying each candidate format in turn
    ## and accepting the first one under which every present value parses.
    for (fmt in DefaultPosixctFormats()) {
      parsed <- as.POSIXct(vals, format = fmt, tz = "UTC")
      if (all(!is.na(parsed[present]))) {
        return(parsed)
      }
    }
    ## Fall back to a pure Date interpretation (values with no time of day).
    parsed <- tryCatch(as.Date(vals), error = function(e) NULL)
    if (!is.null(parsed) && all(!is.na(parsed[present]))) {
      return(parsed)
    }
    return(NULL)
  }

  hist.continuous <- function(x, xlim=NULL, title="", ...) {
    fin <- is.finite(x)
    x <- x[fin]
    if (is.null(xlim)) xlim <- range(x)
    hist(x, xlim=xlim, axes=FALSE, col="lightgray", ylab = "", main=title, ...)
    axis(1)
  }

  plot.all.missing <- function(title) {
    ## A stub plot indicating that all data are missing.
    plot(c(0,0), type = "n", main=title, axes=FALSE)
    text(x=1.5, y=0, lab="MISSING")
  }

  hist.datetime <- function(x, title="", ...) {
    ## Plot the intensity function of a Date or POSIXt variable, in place of a
    ## histogram, so that the density of events over time is visible.
    if (sum(is.finite(as.numeric(x))) < 2) {
      plot.all.missing(title)
      return()
    }
    plot(Intensity(x), main = title, xlab = "", ylab = "", axes = FALSE, ...)
    axis(1)
  }

  hist.factor <- function(x, title="", ...) {
    ## histogram of a factor variable.
    x <- as.factor(x[!is.na(x)])
    counts <- .CollapseTable(table(x), max.factor, alphabetical = TRUE)
    midpoints <- barplot(counts,
                         col="lightgray",
                         main=title,
                         axes = vertical.axes,
                         names.arg = "",
                         ylim = range(c(0, counts)) * 1.02,
                         ...)
    abline(h = 0)
    axis(side = 1, at = as.numeric(midpoints), labels = names(counts),
         lwd = 0, lwd.ticks = 1)
  }

  hist.variable <- function(x, title, xlim, ...){
    datetime <- as.datetime(x)
    if (!is.null(datetime)) {
      hist.datetime(datetime, title, ...)
      if (boxes) box()
      return()
    }
    if (!any(is.finite(x))) {
      plot.all.missing(title)
      return()
    }
    if (is.continuous(x)) {
      hist.continuous(x, xlim, title, ...)
    } else hist.factor(as.factor(x), title, ...)
    if (boxes) box()
  }

  get.range <- function(x){
    nc <- ncol(x)
    xlim <- NULL
    for (i in 1:nc) {
      if (is.continuous(x[,i])){
        fin <- is.finite(x[,i])
        y <- x[fin,i]
        rng <- range(y)
        if (is.null(xlim)) xlim <- rng
        else xlim <- range(c(xlim, rng))
      }
    }
    return(xlim)
  }

  my.margins <- c(3, gap/2, 2, gap/2)
  if (vertical.axes) my.margins[2] <- 2
  original.par <- par(mfrow=c(number.of.rows, number.of.columns),
                      mar = my.margins,
                      oma = rep(2, 4),
                      xpd = !boxes)
  on.exit(par(original.par))

  if (same.scale) xlim <- get.range(x)
  else xlim=NULL

  count <- 0
  for (j in 1 : number.of.rows) {
    for (k in 1 : number.of.columns) {
      count <- count + 1
      if (count > number.of.variables) break
      hist.variable(x[,count], xlim = xlim, title=vnames[count], ...)
      if (vertical.axes) {
        axis(2)
      }
    }
  }
  return(invisible(NULL))
}
