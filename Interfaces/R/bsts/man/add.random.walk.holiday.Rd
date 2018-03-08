\name{add.random.walk.holiday}
\alias{AddRandomWalkHoliday}

\Rdversion{1.1}
\title{
  Random Walk Holiday State Model
}

\description{
 Adds a random walk holiday state model to the state specification.
 This model says

 \deqn{%
   y_t = \alpha_{d(t), t} + \epsilon_t
 }{%
   y[t] = alpha[d(t), t] + observation_error, 
 }

 where there is one element in \eqn{\alpha_t}{alpha[, t]} for each day
 in the holiday influence window.  The transition equation is

 \deqn{ %
   \alpha_{d(t+1), t+1} = \alpha_{d(t+1), t} + \epsilon_{t+1}
 }{ %
   alpha[d(t+1), t+1] = alpha[d(t+1), t] + state_error
   }

 if t+1 occurs on day d(t+1) of the influence window, and

 \deqn{
      \alpha_{d(t+1), t+1} = \alpha_{d(t+1), t} %
 }{%
   alpha[d(t+1), t+1] = alpha[d(t+1), t]%
   }

 otherwise.

}

\usage{
AddRandomWalkHoliday(state.specification = NULL,
                     y,
                     holiday,
                     time0, 
                     sigma.prior = NULL,
                     initial.state.prior = NULL,
                     sdy = sd(as.numeric(y), na.rm = TRUE))
}

\arguments{

  \item{state.specification}{A list of state components that you wish
    augment.  If omitted, an empty list will be assumed.  }

  \item{y}{The time series to be modeled, as a numeric vector
    convertible to \code{\link[xts]{xts}}.  This state model assumes \code{y}
    contains daily data.}

  \item{holiday}{An object of class \code{\link{Holiday}} describing the
  influence window of the holiday being modeled.}

  \item{time0}{An object convertible to \code{\link{Date}} containing
    the date of the initial observation in the training data.  If
    omitted and \code{y} is a \code{\link[zoo]{zoo}} or
    \code{\link[xts]{xts}} object, then \code{time0} will be obtained
    from the index of \code{y[1]}.}

  \item{sigma.prior}{An object created by \code{\link[Boom]{SdPrior}}
    describing the prior distribution for the standard deviation of the
    random walk increments.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the the initial state vector (at time 1).}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

}

\value{

  A list describing the specification of the random walk holiday state
  model, formatted as expected by the underlying C++ code.

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
