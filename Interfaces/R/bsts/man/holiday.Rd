\name{holiday}
\alias{Holiday}
\alias{holiday}
\alias{NamedHoliday}
\alias{FixedDateHoliday}
\alias{NthWeekdayInMonthHoliday}
\alias{LastWeekdayInMonthHoliday}
\alias{DateRangeHoliday}

\Rdversion{1.1}

\title{
  Specifying Holidays
}

\description{
  Specify holidays for use with holiday state models.  
  
}

\usage{

FixedDateHoliday(holiday.name,
                 month = base::month.name,
                 day,
                 days.before = 1,
                 days.after = 1)

NthWeekdayInMonthHoliday(holiday.name,
                         month = base::month.name,
                         day.of.week = weekday.names,
                         week.number = 1,
                         days.before = 1,
                         days.after = 1)

LastWeekdayInMonthHoliday(holiday.name,
                          month = base::month.name,
                          day.of.week = weekday.names,
                          days.before = 1,
                          days.after = 1)

NamedHoliday(holiday.name = named.holidays,
             days.before = 1,
             days.after = 1)

DateRangeHoliday(holiday.name,
                 start.date,
                 end.date)
}

\arguments{

  \item{holiday.name}{A string that can be used to label the holiday in
    output.}

  \item{month}{A string naming the month in which the holiday occurs.
    Unambiguous partial matches are acceptable.  Capitalize the first
    letter.}

  \item{day}{An integer specifying the day of the month on which the
    \code{FixedDateHoliday} occurs.}

  \item{day.of.week}{A string giving the day of the week on which the
    holiday occurs.}

  \item{week.number}{An integer specifying the week of the month on
    which the \code{NthWeekdayInMonthHoliday} occurs.}

  \item{days.before}{An integer giving the number of days of influence
    that the holiday exerts prior to the actual holiday.}

  \item{days.after}{An integer giving the number of days of influence
    that holiday exerts after the actual holiday.}

  \item{named.holidays}{A character vector containing one or more
    recognized holiday names.}

  \item{start.date}{A vector of starting dates for the holiday.  Each
     instance of the holiday in the training data or the forecast period
     must be represented by an element in this vector.  Thus if this is
     an annual holiday and, there are 10 years of training data, and a
     1-year forecast is needed, then this will be a vector of length 11.
   }

   \item{end.date}{A vector of ending dates for the holiday.  Each date
     must occur on or after the corresponding element of
     \code{start.date}, and \code{end.date[i]} must come before
     \code{start.date[i+1]}.
   }
}
   
\value{ Each function returns a list containing the information from the
  function arguments, formatted as expected by the underlying C++ code.
  State models that focus on holidays, such as
  \code{\link{AddRandomWalkHoliday}},
  \code{\link{AddRegressionHoliday}}, and
  \code{\link{AddHierarchicalRegressionHoliday}}, will expect one or
  more holiday objects as arguments.

  \itemize{

    \item{\code{FixedDateHoliday} describes a holiday that occurs on the
      same date each year, like US independence day (July 4).}

    \item{\code{NthWeekdayInMonthHoliday} describes a holiday that
      occurs a particular weekday of a particular week of a particular
      month.  For example, US Labor Day is the first Monday in
      September.}

    \item{\code{LastWeekdayInMonthHoliday} describes a holiday that
      occurs on the last instance of a particular weekday in a
      particular month.  For example, US Memorial Day is the last Monday
      in May.}

    \item{\code{DateRangeHoliday} describes an irregular holiday that
      might not follow a particular pattern.  You can handle this type
      of holiday by manually specifying a range of dates for each
      instance of the holiday in your data set.  NOTE: If you plan on
      using the model to forecast, be sure to include date ranges in the
      forecast period as well as the period covered by the training
      data.  }

    \item{\code{NamedHoliday} is a convenience class for describing
      several important holidays in the US.}
  }
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{AddRandomWalkHoliday}},
  \code{\link{AddRegressionHoliday}},
  \code{\link{AddHierarchicalRegressionHoliday}}
}

\examples{

july4 <- FixedDateHoliday("July4", "July", 4)
memorial.day <- LastWeekdayInMonthHoliday("MemorialDay", "May", "Monday")
labor.day <- NthWeekdayInMonthHoliday("LaborDay", "September", "Monday", 1)
another.way.to.get.memorial.day <- NamedHoliday("MemorialDay")
easter <- NamedHoliday("Easter")
winter.olympics <- DateRangeHoliday("WinterOlympicsSince2000",
                     start = as.Date(c("2002-02-08",
                                        "2006-02-10",
                                        "2010-02-12",
                                        "2014-02-07",
                                        "2018-02-07")),
                     end = as.Date(c("2002-02-24",
                                     "2006-02-26",
                                     "2010-02-28",
                                     "2014-02-23",
                                     "2018-02-25")))

}

