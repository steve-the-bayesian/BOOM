## This file contains utilities for specifying holidays for use in bsts models.
## Holidays are inputs into HolidayStateModels, and are not models themselves.
##
## A holiday is defined as a sequence of intervals defining the start and end of
## the period in which the holiday influences the time series.  Often this can
## be done by specifying a recurring date each year, to anchor the interval, and
## a number of days before and after the holiday date to determine the
## interval's end points.
##
## Some holidays don't occur yearly (e.g. the world cup or the olympics only
## occur every 4 years), and some occur irregularly (e.g. some religious
## holidays require subjective input from religious leaders).

## A global variable naming the days of the week.
## Source: https://www.youtube.com/watch?v=kfVsfOSbJY0
weekday.names <- c("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                   "Friday", "Saturday")

## A list of holidays whose structure is pre-coded.
named.holidays <- c("NewYearsDay",
                    "SuperBowlSunday",
                    "MartinLutherKingDay",
                    "PresidentsDay",
                    "ValentinesDay",
                    "SaintPatricksDay",
                    "USDaylightSavingsTimeBegins",
                    "USDaylightSavingsTimeEnds",
                    "EasterSunday",
                    "USMothersDay",
                    "IndependenceDay",
                    "LaborDay",
                    "ColumbusDay",
                    "Halloween",
                    "Thanksgiving",
                    "MemorialDay",
                    "VeteransDay",
                    "Christmas")

FixedDateHoliday <- function(holiday.name,
                             month = base::month.name,
                             day,
                             days.before = 1,
                             days.after = 1) {
  ## For specifying holidays that occur on the same date each year, like US
  ## independence day (July 4).
  ## Args:
  ##   holiday.name:  A string giving the name of this holiday.
  ##   month: A string giving the name of the month in which this holiday
  ##     occurs.  
  ##   day: A number giving the day of the month on which the holiday occurs.
  ##   days.before: The expected number of days before the holiday date that the
  ##     holiday will affect the time series,
  ##   days.before: The expected number of days after the holiday date that the
  ##     holiday will affect the time series,
  ##
  ## Returns:
  ##   A list containing the information in the arguments, properly formatted
  ##   for passing to C++ code.
  stopifnot(is.character(holiday.name), length(holiday.name) == 1)
  month <- match.arg(month)
  stopifnot(is.numeric(day), length(day) == 1, day >= 1, day <= 31)
  if (month %in% c("September", "April", "June", "November")) {
    stopifnot(day <= 30)
  } else if (month == "February") {
    stopifnot(day <= 28)
  }
  
  
  stopifnot(is.numeric(days.before), length(days.before) == 1, days.before >= 0)
  stopifnot(is.numeric(days.after), length(days.after) == 1, days.after >= 0)

  ans <- list(name = holiday.name,
              month = month,
              day = as.integer(day),
              days.before = as.integer(days.before),
              days.after = as.integer(days.after))
  class(ans) <- c("FixedDateHoliday", "Holiday")
  return(ans)
}

NamedHoliday <- function(holiday.name = named.holidays,
                         days.before = 1,
                         days.after = 1) {
  ## Args:
  ##   holiday.name: The name of the holiday.  This must be from a specified
  ##     list of holidays about which the underlying C++ code is aware.
  ##   days.before: The expected number of days before the holiday date that the
  ##     holiday will affect the time series,
  ##   days.before: The expected number of days after the holiday date that the
  ##     holiday will affect the time series,
  ##
  ## Returns:
  ##   A list containing the information in the arguments, properly formatted
  ##   for passing to C++ code.
  holiday.name <- match.arg(holiday.name)
  stopifnot(is.numeric(days.before), length(days.before) == 1, days.before >= 0)
  stopifnot(is.numeric(days.after), length(days.after) == 1, days.after >= 0)
  ans <- list(name = holiday.name,
              days.before = as.integer(days.before),
              days.after = as.integer(days.after))
  class(ans) <- c("NamedHoliday", "Holiday")
  return(ans)
}

NthWeekdayInMonthHoliday <- function(holiday.name,
                                     month = base::month.name,
                                     day.of.week = weekday.names,
                                     week.number = 1,
                                     days.before = 1,
                                     days.after = 1) {
  ## For specifying holidays like US Labor Day, which is the first Monday in
  ## September.
  ##
  ## Args:
  ##   holiday.name:  A string giving the name of this holiday.
  ##   month: A string giving the name of the month in which this holiday
  ##     occurs.  
  ##   day.of.week: A string naming the day of the week on which this holiday
  ##     occurs.
  ##   week.number: An integer giving the week of the month on which this
  ##     holiday occurs.
  ##   days.before: The expected number of days before the holiday date that the
  ##     holiday will affect the time series,
  ##   days.before: The expected number of days after the holiday date that the
  ##     holiday will affect the time series,
  ##
  ## Returns:
  ##   A list containing the information in the arguments, properly formatted
  ##   for passing to C++ code.
  stopifnot(is.character(holiday.name), length(holiday.name) == 1)
  month <- match.arg(month)
  day.of.week <- match.arg(day.of.week)
  stopifnot(is.numeric(week.number), length(week.number) == 1)
  week.number <- as.integer(week.number);
  stopifnot(is.numeric(days.before), length(days.before) == 1, days.before >= 0)
  stopifnot(is.numeric(days.after), length(days.after) == 1, days.after >= 0)

  ans <- list(name = holiday.name,
              month = month,
              day.of.week = day.of.week,
              week.number = as.integer(week.number),
              days.before = as.integer(days.before),
              days.after = as.integer(days.after))
  class(ans) <- c("NthWeekdayInMonthHoliday", "Holiday")
  return(ans)
}

LastWeekdayInMonthHoliday <- function(holiday.name,
                                      month = base::month.name,
                                      day.of.week = weekday.names,
                                      days.before = 1,
                                      days.after = 1) {
  ## For specifying holidays like US Memorial Day, which is the last Monday in
  ## May.
  ##
  ## Args:
  ##   holiday.name:  A string giving the name of this holiday.
  ##   month: A string giving the name of the month in which this holiday
  ##     occurs.  
  ##   day.of.week: A string naming the day of the week on which this holiday
  ##     occurs.
  ##   days.before: The expected number of days before the holiday date that the
  ##     holiday will affect the time series,
  ##   days.before: The expected number of days after the holiday date that the
  ##     holiday will affect the time series,
  ##
  ## Returns:
  ##   A list containing the information in the arguments, properly formatted
  ##   for passing to C++ code.
  stopifnot(is.character(holiday.name), length(holiday.name) == 1)
  month <- match.arg(month)
  day.of.week <- match.arg(day.of.week)
  stopifnot(is.numeric(days.before), length(days.before) == 1, days.before >= 0)
  stopifnot(is.numeric(days.after), length(days.after) == 1, days.after >= 0)

  ans <- list(name = holiday.name,
              month = month,
              day.of.week = day.of.week,
              days.before = as.integer(days.before),
              days.after = as.integer(days.after))
  class(ans) <- c("LastWeekdayInMonthHoliday", "Holiday")
  return(ans)
}

DateRangeHoliday <- function(holiday.name,
                             start.date,
                             end.date) {
  ## For manually specifying an irregular holiday through the start and end of
  ## its period of influence.
  ##
  ## Args:
  ##   holiday.name:  A string giving the name of this holiday.
  ##   start.date: A vector of dates giving the start date for each holiday
  ##     occurrance.  Must be coercible to class Date using as.Date.
  ##   end.date: A vector of dates giving the start date for each holiday
  ##     occurrance.  This object must be coercible to class Date using as.Date,
  ##     its length must match that of 'start.date', and each element in
  ##     'end.date' must occur after the corresponding element of 'start.date',
  ##     and before the next element in 'start.date'.
  ##
  ## NOTE: If you plan on using this model for prediction, be sure to include
  ##       date ranges that cover the prediction period as well as covering the
  ##       training data.
  ##
  ## Returns:
  ##   A list containing the information in the arguments, properly formatted
  ##   for passing to C++ code.
  stopifnot(is.character(holiday.name), length(holiday.name) == 1)
  start.date <- sort(as.Date(start.date))
  end.date <- sort(as.Date(end.date))
  ## Check that there are the same number of start dates and end dates, that
  ## start.date is always before end date, and that the next start date always
  ## happens after the end of this period's end date.
  stopifnot(length(start.date) == length(end.date),
            all(start.date <= end.date),
            all(tail(start.date, -1) > head(end.date, -1)))
  ans <- list(name = holiday.name,
              start.date = start.date,
              end.date = end.date)
  class(ans) <- c("DateRangeHoliday", "Holiday")
  return(ans)
}

MaxWindowWidth <- function(holiday, ...) {
  ## Return the maximum influence window width of the given holiday.
  stopifnot(inherits(holiday, "Holiday"))
  UseMethod("MaxWindowWidth")
}

MaxWindowWidth.default <- function(holiday, ...) {
  ## Handles the typical case where a holiday occur on a specific day, and has
  ## days.before and days.after components.
  return(1 + holiday$days.before + holiday$days.after)
}

MaxWindowWidth.DateRangeHoliday <- function(holiday, ...) {
  ## Returns the window width for a DateRangeHoliday.
  dt <- 1 + difftime(holiday$start, holiday$end, units = "days")
  return(as.numeric(max(dt)))
}

DateRange <- function(holiday, timestamps) {
  ## Returns the first and last dates of the influence window for the given
  ## holiday, among the given timestamps.
  ##
  ## Args:
  ##   holiday:  An object inheriting from class "Holiday".
  ##   timestamps:  A vector of class Date or POSIXt.
  ##
  ## NOTE: This function was written with the expectation that timestamps
  ## contain daily data.  Use with caution otherwise.
  ##
  ## Returns:
  ##   A two-column data frame giving the start and end times for each incidence
  ##   of the holiday during the period spanned by timestamps.
  if (!inherits(timestamps, "Date")
    && !inherits(timestamps, "POSIXt")) {
    stop("Model must have timestamps of class Date or POSIXt.")
  }
  if (inherits(timestamps, "POSIXt")) {
    timestamps <- as.Date(timestamps)
  }
  date.ranges <- .Call("analysis_common_r_get_date_ranges_",
    holiday,
    timestamps,
    PACKAGE = "bsts");
  start <- as.Date(timestamps[date.ranges[, 1]])
  end <- as.Date(timestamps[date.ranges[, 2]])
  return(data.frame(start = start, end = end))
}

