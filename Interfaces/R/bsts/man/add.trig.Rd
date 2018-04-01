\name{add.trig}
\alias{AddTrig}

\Rdversion{1.1}
\title{
  Trigonometric Seasonal State Component
}

\description{
  Add a trigonometric seasonal model to a state specification.
}

\usage{
  AddTrig(
     state.specification = NULL,
     y,
     period,
     frequencies,
     sigma.prior = NULL,
     initial.state.prior = NULL,
     sdy = sd(y, na.rm = TRUE),
     method = c("harmonic", "direct"))
}

\arguments{

  \item{state.specification}{A list of state components that you wish to
    add to.  If omitted, an empty list will be assumed.  }

  \item{y}{ The time series to be modeled, as a numeric vector.}

  \item{period}{A positive scalar giving the number of time steps
    required for the longest cycle to repeat.}

  \item{frequencies}{A vector of positive real numbers giving the number
     of times each cyclic component repeats in a period.  One sine and
     one cosine term will be added for each frequency.  
   }

  \item{sigma.prior}{An object created by \code{\link[Boom]{SdPrior}}
    describing the prior distribution for the standard deviation of the
    increments for the harmonic coefficients.}

  \item{initial.state.prior}{An object created using
    \code{\link[Boom]{NormalPrior}}, describing the prior distribution
    of the the initial state vector (at time 1).}

  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }

  \item{method}{The method of including the sinusoids.  The "harmonic"
    method is strongly preferred, with "direct" offered mainly for
    teaching purposes.  }
}

\value{Returns a list with the elements necessary to specify a seasonal
  state model.}

\details{

  \subsection{Harmonic Method}{ 
    Each frequency \eqn{lambda_j = 2\pi j / S}{2 * pi * j / S} where S
    is the period (number of time points in a full cycle) is associated
    with two time-varying random components: \eqn{\gamma_{jt}}{gamma[j,
    t]}, and \eqn{gamma^*_{jt}}{gamma^*[j, t]}.  They evolve through
    time as

   \deqn{%
     \gamma_{j, t + 1} = \gamma_{jt}  \cos(\lambda_j) + \gamma^*_{j, t}  %
     \sin(\lambda_j) + \epsilon_{0t}}{%
     gamma[j, t + 1] = \gamma[j, t] * cos(\lambda_j)%
                     - \gamma^*[j, t] * sin(lambda_j) + error_0}
    \deqn{%
      \gamma^*_{j, t + 1} = \gamma^*[j, t] \cos(\lambda_j) - \gamma_{jt} *
      \sin(\lambda_j) + \epsilon_1}{%
      gamma^*[j, t + 1] = \gamma^*[j, t] * cos(\lambda_j)
                        - \gamma[j, t] * sin(lambda_j) + error_1}


  where \eqn{\epsilon_0}{error_0} and \eqn{\epsilon_1}{error_1} are
  independent with the same variance.  This is the real-valued version
  of a harmonic function: \eqn{\gamma \exp(i\theta)}{gamma * exp(i * theta)}.
   
  The transition matrix multiplies the function by
  \eqn{\exp(i \lambda_j}{exp(i * lambda_j)}, so that
  after 't' steps the harmonic's value is
  \eqn{\gamma \exp(i \lambda_j t)}{gamma * exp(i * lambda * t)}.

  The model dynamics  allows gamma to drift over time in a random walk.
  
  The state of the model is
  \eqn{(\gamma_{jt}, \gamma^*_{jt})}{(gamma_jt, gamma^*_jt)},
  for j = 1, ... number of frequencies.

  The state transition matrix is a block diagonal matrix, where block 'j' is

  \deqn{\cos(\lambda_j)   \sin(\lambda_j)}{cos(lambda_j)   sin(lambda_j)}
  \deqn{-\sin(\lambda_j)  \cos(\lambda_j)}{-sin(lambda_j)   cos(lambda_j)}

  The error variance matrix is sigma^2 * I.  There is a common sigma^2
  parameter shared by all frequencies.

  The model is full rank, so the state error expander matrix R_t is the
  identity.
  
  The observation_matrix is (1, 0, 1, 0, ...), where the 1's pick out the
  'real' part of the state contributions.
}

\subsection{Direct Method}{
  Under the 'direct' method the trig component adds a collection of sine
  and cosine terms with randomly varying coefficients to the state
  model.  The coefficients are the states, while the sine and cosine
  values are part of the "observation matrix".

  This state component adds the sum of its terms to the observation
  equation.

  \deqn{y_t = \sum_j \beta_{jt} sin(f_j t) + \gamma_{jt} cos(f_j t)}{ %
    y_t = beta[1, t] * sin(f[1] * t) + ... + beta[F, t] * sin(f[F] * t)
        + gamma[j, t] * cos(f[1] * t) + ... + gamma[F, t] * cos(f[F] * t)
      }

  The evolution equation is that each of the sinusoid coefficients
  follows a random walk with standard deviation sigma[j].

  \deqn{\beta_{jt} = \beta_{jt-1} + N(0, sigma_{sj}^2)
    \gamma_{jt} = \gamma_{j-1} + N(0, sigma_{cj}^2) }{%
    beta[j, t] = beta[j, t-1] + N(0, sigma[j, 1])^2
    gamma[j, t] = gamma[j, t-1] + N(0, sigma[j, 2])^2
  }

  The direct method is generally inferior to the harmonic method.  It
  may be removed in the future.
}
}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
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
  ss <- AddTrig(ss, y, period = 12, frequencies = 1:3)
  model <- bsts(y, state.specification = ss, niter = 200)
  plot(model)

  ## The "harmonic" method is much more stable than the "direct" method.
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddTrig(ss, y, period = 12, frequencies = 1:3, method = "direct")
  model2 <- bsts(y, state.specification = ss, niter = 200)
  plot(model2)

}

\keyword{models}
