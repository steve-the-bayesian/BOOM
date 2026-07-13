\name{DetectPosixct}
\Rdversion{1.1}
\alias{DetectPosixct}
\alias{ConvertPosixct}
\alias{DefaultPosixctFormats}
\title{
  Detect and convert date-time columns in a data frame
}

\description{
  Utilities for finding character or factor columns in a data frame that hold
  date-time (\code{\link{POSIXct}}) information, and for converting them to
  \code{POSIXct} in place.
}

\usage{
DetectPosixct(df, n = 100, formats = DefaultPosixctFormats(),
              tz = "UTC", min.frac = 1, exclude = NULL)

ConvertPosixct(dframe, tz = "UTC", ...)

DefaultPosixctFormats()
}

\arguments{

  \item{df, dframe}{
    The data frame to examine or convert.
  }

  \item{n}{
    The number of leading rows to inspect.  Restricting attention to the head
    of the frame keeps the check inexpensive on large data sets.
  }

  \item{formats}{
    A character vector of candidate date-time formats, in the notation used by
    \code{\link{strptime}}.  Each column's values are tried against these
    formats in order.  Every format supplied by \code{DefaultPosixctFormats}
    carries a time component, so pure \code{\link{Date}} columns are not
    flagged.
  }

  \item{tz}{
    The time zone assumed when parsing values.  In \code{ConvertPosixct} it is
    also passed to \code{DetectPosixct} so that detection and conversion agree.
  }

  \item{min.frac}{
    The minimum fraction of non-missing values in a column that must parse
    successfully for the column to be flagged.  A value of 1 (the default)
    requires that every non-missing value parse.
  }

  \item{exclude}{
    An optional character vector of column names to leave alone.  Named columns
    are never flagged, even if their values parse as date-times.
  }

  \item{\dots}{
    Extra arguments passed from \code{ConvertPosixct} to \code{DetectPosixct}
    (e.g. \code{n}, \code{formats}, \code{min.frac}, or \code{exclude}).
  }

}

\value{
  \code{DetectPosixct} returns a two-column character matrix with one row per
  column that should be classified as \code{POSIXct}.  The \code{"variable"}
  column gives the name of the flagged column, and the \code{"format"} column
  gives the date-time format string to pass to \code{\link{as.POSIXct}}.  The
  matrix has zero rows if no such column is found.

  \code{ConvertPosixct} returns a copy of \code{dframe} in which every detected
  date-time column has been replaced by a \code{POSIXct} column of the same
  name.  Columns that are not detected as date-times are returned unchanged.

  \code{DefaultPosixctFormats} returns the character vector of candidate
  date-time formats used by \code{DetectPosixct} when no explicit
  \code{formats} argument is supplied.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{as.POSIXct}},
  \code{\link{strptime}}.
}

\examples{
df <- data.frame(
  id = 1:3,
  event.time = c("2020-01-01 08:30:00",
                 "2020-01-02 09:15:00",
                 "2020-01-03 10:00:00"),
  label = c("a", "b", "c"),
  stringsAsFactors = FALSE)

## Identify the date-time column(s) and the format to use.
DetectPosixct(df)

## Convert them in place.
converted <- ConvertPosixct(df)
sapply(converted, class)

## Inspect (or override) the default candidate formats.
DefaultPosixctFormats()
}

\keyword{ chron }
\keyword{ manip }
