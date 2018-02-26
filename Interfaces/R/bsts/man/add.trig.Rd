\name{add.trig}
\alias{AddTrig}

\Rdversion{1.1}
\title{
  Trigonometric seasonal state component
}

\description{
  Add a trigonometric seasonal model to a state specification.

  The trig component adds a collection of sine and cosine terms with
  randomly varying coefficients to the state model.  The coefficients
  are the states, while the sine and cosine values are part of the
  "observation matrix".

  This state component adds the sum of its terms to the observation
  equation.

  \deqn{y_t = \sum_j \beta_{jt} sin(f_j t) + \gamma_{jt} cos(f_j t)}{ %
    y_t = beta[1, t] * sin(f[1] * t) + ... + beta[F, t] * sin(f[F] * t)
        + gamma[j, t] * cos(f[1] * t) + ... + gamma[F, t] * cos(f[F] * t)
      }

  The evolution equation is that each of the sinusoid coefficients
  follows a random walk with standard deviation sigma[j].

  \deqn{\beta_{jt} = \beta_{jt-1} + N(0, sigma_{sj}^2))
    \gamma_{jt} = \gamma_{j-1} + N(0, sigma_{cj}^2) }{%
    beta[j, t] = beta[j, t-1] + N(0, sigma[j, 1])^2
    gamma[j, t] = gamma[j, t-1] + N(0, sigma[j, 2])^2
}
}

\usage{
  AddTrig(
     state.specification = NULL,
     y,
     period,
     frequencies,
     sigma.prior = NULL,
     initial.state.prior = NULL,
     sdy)
}

\arguments{

  \item{state.specification}{A list of state components that you wish to
    add to.  If omitted, an empty list will be assumed.  }

  \item{y}{ The time series to be modeled, as a numeric vector.}

  \item{period}{A positive scalar giving the number of time steps
    required for the longest cycle to repeat.}

  \item{frequencies}{A vector of positive real numbers giving the number
     of times each cyclic component repeats in a period.  One sine and
     one cosine term will be added for each frequency.}

  \item{sigma.prior}{An object created by \code{\link[Boom]{SdPrior}}
    describing the prior distribution for the standard deviation of
    coefficients of the sinusoid terms.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the the initial state vector (at time 1).}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

}

\value{Returns a list with the elements necessary to specify a seasonal
  state model.}

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
  \code{\link[Boom]{MvnPrior}}
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddTrig(ss, y, period = 12, frequencies = 1)
  model <- bsts(y, state.specification = ss, niter = 100)
  plot(model)
}

\keyword{models}
