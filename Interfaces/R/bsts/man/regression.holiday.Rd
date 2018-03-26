\name{regression.holiday}
\alias{AddRegressionHoliday}
\alias{RegressionHolidayStateModel}
\alias{AddHierarchicalRegressionHoliday}
\alias{HierarchicalRegressionHolidayStateModel}

\Rdversion{1.1}
\title{
  Regression Based Holiday Models
}

\description{
  Add a regression-based holiday model to the state specification.
}

\details{

  The model assumes that
  \deqn{
    y_t  = \beta_{d(t)} + \epsilon_t %
  }{
    y[t] = beta[d(t)] + observation_error
  }

  The regression state model assumes vector of regression coefficients
  \eqn{\beta}{beta} contains elements \eqn{\beta_d \sim N(0,
    \sigma)}{beta[d] ~ N(0, sigma)}.

  The HierarchicalRegressionHolidayModel assumes \eqn{\beta}{beta} is
  composed of holiday-specific sub-vectors
  \eqn{\beta_h \sim N(b_0, V)}{beta[h, ] ~ N(b0, V)}, where each
  \eqn{\beta_h}{beta[h,]} contains coefficients describing the days in
  the influence window of holiday h.  The hierarchical version of the
  model treats \eqn{b_0}{b0} and \eqn{V}{V} as parameters to be learned,
  with prior distributions
  \deqn{b_0 \sim N(\bar b, \Omega)}{b0 ~ N(b.bar, Omega)} and
  \deqn{V \sim IW(\nu, S)}{V ~ IW(nu, S).}
  where \eqn{IW}{IW} represents the inverse Wishart distribution.
}

\usage{

AddRegressionHoliday(
    state.specification = NULL,
    y,
    holiday.list,
    time0 = NULL,
    prior = NULL,
    sdy = sd(as.numeric(y), na.rm = TRUE))

AddHierarchicalRegressionHoliday(
    state.specification = NULL,
    y,
    holiday.list,
    coefficient.mean.prior = NULL,
    coefficient.variance.prior = NULL,
    time0 = NULL,
    sdy = sd(as.numeric(y), na.rm = TRUE))
}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{holiday.list}{A list of objects of type \code{\link{Holiday}}.
    The width of the influence window should be the same number of days
    for all the holidays in this list.  If the data contains many
    instances of holidays with different window widths, then multiple
    instances HierarchicalRegressionHolidayModel can be used as long as
    all holidays in the same state component model have the same sized
    window width.}
  
  \item{y}{The time series to be modeled, as a numeric vector
    convertible to \code{\link[xts]{xts}}.  This state model assumes \code{y}
    contains daily data.}

  \item{prior}{An object of class \code{\link[Boom]{NormalPrior}}
    describing the expected variation among daily holiday effects.}

  \item{coefficient.mean.prior}{An object of type
    \code{\link[Boom]{MvnPrior}} giving the hyperprior for the average
    effect of a holiday in each day of the influence window.}

  \item{coefficient.variance.prior}{An object of type
    \code{\link[Boom]{InverseWishartPrior}} describing the prior belief
    about the variation in holiday effects from one holiday to the
    next.}

  \item{time0}{An object convertible to \code{\link{Date}} containing
    the date of the initial observation in the training data.  If
    omitted and \code{y} is a \code{\link[zoo]{zoo}} or
    \code{\link[xts]{xts}} object, then \code{time0} will be obtained
    from the index of \code{y[1]}.}
  
  \item{sdy}{The standard deviation of the series to be modeled.  This
    will be ignored if \code{y} is provided, or if all the required
    prior distributions are supplied directly.  }
}

\value{ Returns a list with the elements necessary to specify a local
  linear trend state model.}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
trend <- cumsum(rnorm(730, 0, .1))
  dates <- seq.Date(from = as.Date("2014-01-01"), length = length(trend), by = "day")
  y <- zoo(trend + rnorm(length(trend), 0, .2), dates)

AddHolidayEffect <- function(y, dates, effect) {
  ## Adds a holiday effect to simulated data.
  ## Args:
  ##   y: A zoo time series, with Dates for indices.
  ##   dates: The dates of the holidays.
  ##   effect: A vector of holiday effects of odd length.  The central effect is
  ##     the main holiday, with a symmetric influence window on either side.
  ## Returns:
  ##   y, with the holiday effects added.
  time <- dates - (length(effect) - 1) / 2
  for (i in 1:length(effect)) {
    y[time] <- y[time] + effect[i]
    time <- time + 1
  }
  return(y)
}

## Define some holidays.
memorial.day <- NamedHoliday("MemorialDay")
memorial.day.effect <- c(.3, 3, .5)
memorial.day.dates <- as.Date(c("2014-05-26", "2015-05-25"))
y <- AddHolidayEffect(y, memorial.day.dates, memorial.day.effect)

presidents.day <- NamedHoliday("PresidentsDay")
presidents.day.effect <- c(.5, 2, .25)
presidents.day.dates <- as.Date(c("2014-02-17", "2015-02-16"))
y <- AddHolidayEffect(y, presidents.day.dates, presidents.day.effect)

labor.day <- NamedHoliday("LaborDay")
labor.day.effect <- c(1, 2, 1)
labor.day.dates <- as.Date(c("2014-09-01", "2015-09-07"))
y <- AddHolidayEffect(y, labor.day.dates, labor.day.effect)

## The holidays can be in any order.
holiday.list <- list(memorial.day, labor.day, presidents.day)

## In a real example you'd want more than 100 MCMC iterations.
niter <- 100

## Fit the model
ss <- AddLocalLevel(list(), y)
ss <- AddRegressionHoliday(ss, y, holiday.list = holiday.list)
model <- bsts(y, state.specification = ss, niter = niter)

## Plot all model state components.
plot(model, "comp")

## Plot the specific holiday state component.
plot(ss[[2]], model)

## Try again with some shrinkage.  With only 3 holidays there won't be much
## shrinkage.  
ss2 <- AddLocalLevel(list(), y)

## Plot the specific holiday state component.
ss2 <- AddHierarchicalRegressionHoliday(ss2, y, holiday.list = holiday.list)
model2 <- bsts(y, state.specification = ss2, niter = niter)

plot(model2, "comp")
plot(ss2[[2]], model2)
}

\seealso{
  \code{\link{bsts}}.
  \code{\link{RandomWalkHolidayStateModel}}.
  \code{\link[Boom]{SdPrior}}
  \code{\link[Boom]{NormalPrior}}
}

\keyword{models}
