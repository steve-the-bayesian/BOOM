\name{add.holiday}
\alias{AddNamedHolidays}
\alias{NamedHolidays}
\alias{AddFixedDateHoliday}
\alias{AddNthWeekdayInMonthHoliday}
\alias{AddLastWeekdayInMonthHoliday}

\Rdversion{1.1}
\title{
  Holiday state models
}

\description{


  Add a random-walk holiday model to a state specification.

  This model allows each active day in a holiday window to move
  according to a random walk relative to the day's value the last time
  the holiday occurred.

}

\usage{

AddFixedDateHoliday(state.specification = NULL,
                    holiday.name,
                    month,
                    day,
                    y,
                    sigma.prior = NULL,
                    initial.state.prior = NULL,
                    sdy = sd(as.numeric(y), na.rm = TRUE),
                    time0 = NULL,
                    days.before = 1,
                    days.after = 1)

AddNthWeekdayInMonthHoliday(state.specification = NULL,
                            holiday.name,
                            month,
                            day.of.week,
                            which.week,
                            y,
                            sigma.prior = NULL,
                            initial.state.prior = NULL,
                            sdy = sd(as.numeric(y), na.rm = TRUE),
                            time0 = NULL,
                            days.before = 1,
                            days.after = 1)

AddLastWeekdayInMonthHoliday(state.specification = NULL,
                             holiday.name,
                             month,
                             day.of.week,
                             y,
                             sigma.prior = NULL,
                             initial.state.prior = NULL,
                             sdy = sd(as.numeric(y), na.rm = TRUE),
                             time0 = NULL,
                             days.before = 1,
                             days.after = 1)

NamedHolidays(except = NULL)

AddNamedHolidays(state.specification = NULL,
                 named.holidays = NamedHolidays(),
                 y,
                 sigma.prior = NULL,
                 initial.state.prior = NULL,
                 sdy = sd(as.numeric(y), na.rm = TRUE),
                 time0 = NULL,
                 days.before = 1,
                 days.after = 1)
}

\arguments{

  \item{state.specification}{A list of state components that you wish
    augment.  If omitted, an empty list will be assumed.  }

  \item{holiday.name}{A string that can be used to label the holiday in
    output.}

  \item{named.holidays}{A character vector containing one or more
    recognized holiday names.}

  \item{y}{The time series to be modeled, as a numeric vector
    convertible to \code{\link[xts]{xts}}.  This state model assumes \code{y}
    contains daily data.}

  \item{sigma.prior}{An object created by \code{\link[Boom]{SdPrior}}
    describing the prior distribution for the standard deviation of the
    random walk increments.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the the initial state vector (at time 1).}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

  \item{time0}{An object convertible to \code{\link{POSIXt}} containing
    the date of the initial observation in the training data.  If
    omitted and \code{y} is a \code{\link[zoo]{zoo}} or \code{\link[xts]{xts}}
    object, then \code{time0} will be obtained from the index of
    \code{y[1]}.}

  \item{days.before}{An integer giving the number of days the influence
       of the holiday extends prior to the actual holiday.}

  \item{days.after}{An integer giving the number of days the influence
    of the holiday extends after the actual holiday.}

  \item{month}{A string naming the month in which the holiday occurs.
    Unambiguous partial matches are acceptable.  Capitalize the first
    letter.}

  \item{day}{An integer specifying the day of the month on which the
    \code{FixedDateHoliday} occurs.}

  \item{day.of.week}{A string giving the weekday on which the
    \code{NthWeekdayInMonthHoliday} occurs.}

  \item{which.week}{An integer specifying the week of the month on which
    the \code{NthWeekdayInMonthHoliday} occurs.  If \code{which.week <=
    0} then the holiday is assumed to occur on the last
    \code{day.of.week} in \code{month}.  }

  \item{except}{If \code{NULL} then all named holidays are returned.  If
    except is a character vector containing partial matches to holiday
    names, the matched names will be omitted from the returned list.}
 }

\value{

\code{NamedHolidays} returns a character vector with the names of the
  recognized holiday.

The other functions return \code{state.specification}, after adding the
requested state components.

\code{AddNthWeekdayInMonthHoliday}, \code{AddLastWeekdayInMonthHoliday},
and \code{AddFixedDateHoliday} can each add one holiday at a time.
\code{AddNamedHolidays} will add several holidays at once, if
\code{named.holidays} is a vector.

}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\seealso{
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
  \code{\link[Boom]{NormalPrior}}
}

\examples{
## TODO(stevescott):  add examples
}

\keyword{models}
