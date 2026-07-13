DefaultPosixctFormats <- function() {
  ## The set of candidate date-time formats tried by 'DetectPosixct' when no
  ## explicit list of formats is supplied.  Each format carries a time
  ## component, so that pure Date columns (with no time of day) are not
  ## mistakenly flagged as POSIXct.
  ##
  ## Returns:
  ##   A character vector of date-time formats, in the notation used by
  ##   'strptime'.
  return(c(
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y/%m/%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M"
  ))
}

DetectPosixct <- function(df,
                          n = 100,
                          formats = DefaultPosixctFormats(),
                          tz = "UTC",
                          min.frac = 1,
                          exclude = NULL) {
  ## Examine the first several rows of a data frame and identify the
  ## character or factor columns whose values should be classified as
  ## POSIXct (date-time).  A column is flagged when a sufficient
  ## fraction of its non-missing values parse cleanly against one of a
  ## set of candidate date-time formats.
  ##
  ## Args:
  ##   df:  The data frame to examine.
  ##   n:  The number of leading rows to inspect.  Restricting attention
  ##     to the head of the frame keeps the check inexpensive on large
  ##     data sets.
  ##   formats:  A character vector of candidate date-time formats, in
  ##     the notation used by strptime.  Each column's values are tried
  ##     against these formats in order.  Every format carries a time
  ##     component, so pure Date columns are not flagged.
  ##   tz:  The time zone assumed when parsing values.
  ##   min.frac:  The minimum fraction of non-missing values in a column
  ##     that must parse successfully for the column to be flagged.  A
  ##     value of 1 requires that every non-missing value parse.
  ##   exclude:  An optional character vector of column names to leave
  ##     alone.  Named columns are never flagged, even if their values
  ##     parse as date-times.
  ##
  ## Returns:
  ##   A two-column character matrix, with one row per column that should
  ##   be classified as POSIXct.  The first column, "variable", gives the
  ##   name of the flagged column.  The second column, "format", gives the
  ##   date-time format string that should be passed to as.POSIXct to
  ##   perform the conversion.  The matrix has zero rows if no such column
  ##   is found.
  stopifnot(is.data.frame(df))

  head.df <- utils::head(df, n)

  posixct.format <- function(x) {
    ## Determine the single format that should be used to convert this
    ## column to POSIXct, returning NA_character_ when the column should
    ## not be reclassified.  Columns that are already date-times need no
    ## reclassification, and only text-valued columns are candidates for
    ## conversion.
    if (inherits(x, "POSIXct")) {
      return(NA_character_)
    }
    if (!is.character(x) && !is.factor(x)) {
      return(NA_character_)
    }

    vals <- as.character(x)
    vals <- vals[!is.na(vals) & trimws(vals) != ""]
    if (length(vals) == 0) {
      return(NA_character_)
    }

    ## Try each format in turn and return the first one that parses a
    ## sufficient fraction of the values on its own.  A single format is
    ## needed so the column can be converted with one call to as.POSIXct.
    for (fmt in formats) {
      parsed <- as.POSIXct(vals, format = fmt, tz = tz)
      if (mean(!is.na(parsed)) >= min.frac) {
        return(fmt)
      }
    }

    NA_character_
  }

  detected <- vapply(head.df, posixct.format, character(1))

  ## Honor the caller's request to leave specified columns untouched.
  if (length(exclude) > 0) {
    detected[names(detected) %in% exclude] <- NA_character_
  }

  flagged <- !is.na(detected)
  result <- cbind(variable = names(head.df)[flagged],
                  format = unname(detected[flagged]))
  return(result)
}

ConvertPosixct <- function(dframe, tz = "UTC", ...) {
  ## Reclassify the date-time columns of a data frame as POSIXct.
  ##
  ## Args:
  ##   dframe:  A data frame.
  ##   tz:  The time zone assumed when parsing values.  This is also passed
  ##     to DetectPosixct so that detection and conversion agree.
  ##   ...:  Extra arguments passed to DetectPosixct (e.g. 'n', 'formats',
  ##     'min.frac', or 'exclude').
  ##
  ## Returns:
  ##   A copy of dframe, where any character or factor columns containing
  ##   date-time information are replaced by a POSIXct column of the same
  ##   name.  Columns that are not detected as date-times are returned
  ##   unchanged.
  datetime.fields <- DetectPosixct(dframe, tz = tz, ...)
  for (i in seq_len(nrow(datetime.fields))) {
    vname <- datetime.fields[i, "variable"]
    date.format <- datetime.fields[i, "format"]
    dframe[[vname]] <- as.POSIXct(as.character(dframe[[vname]]),
                                  format = date.format,
                                  tz = tz)
  }
  return(dframe)
}
