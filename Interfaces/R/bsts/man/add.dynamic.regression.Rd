% Copyright 2018 Google Inc. All Rights Reserved.
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public
% License along with this library; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

\name{add.dynamic.regression}
\alias{AddDynamicRegression}
\alias{DynamicRegressionOptions}
\alias{DynamicRegressionArOptions}
\alias{DynamicRegressionRandomWalkOptions}

\Rdversion{1.1}
\title{
  Dynamic Regression State Component
}

\description{ Add a dynamic regression component to the state
  specification of a bsts model.  A dynamic regression is a regression
  model where the coefficients change over time according to a random
  walk.  }

\usage{
  AddDynamicRegression(
     state.specification,
     formula,
     data,
     model.options = NULL,
     sigma.mean.prior.DEPRECATED = NULL,
     shrinkage.parameter.prior.DEPRECATED = GammaPrior(a = 10, b = 1),
     sigma.max.DEPRECATED = NULL,
     contrasts = NULL,
     na.action = na.pass)

 DynamicRegressionRandomWalkOptions(sdy = NULL,
                                    sigma.mean.prior = NULL,
                                    shrinkage.parameter.prior =
                                    GammaPrior(a = 10, b = 1),
                                    sigma.max = NULL)

 DynamicRegressionArOptions(lags = 1, sigma.prior = SdPrior(1, 1))
}

\arguments{
  \item{state.specification}{A list of state components that you wish to add to.  If
    omitted, an empty list will be assumed.  }

  \item{formula}{A formula describing the regression portion of the
    relationship between y and X. If no regressors are desired
    then the formula can be replaced by a numeric vector giving
    the time series to be modeled.}

  \item{data}{ An optional data frame, list or environment (or object
    coercible by \code{\link{as.data.frame}} to a data frame) containing
    the variables in the model.  If not found in \code{data}, the
    variables are taken from 'environment(formula)', typically the
    environment from which \code{AddDynamicRegression} is called.  }

  \item{model.options}{An object inheriting from
    \code{\link{DynamicRegressionOptions}} giving the specific transition model for
    the dynamic regression coefficients, and the prior distribution for
    any hyperparameters associated with the transition model.}

  \item{sigma.mean.prior}{ An object created by
    \code{\link[Boom]{GammaPrior}} describing the prior distribution of
    b/a (see details below).}

  \item{sigma.mean.prior.DEPRECATED}{ This option should be set using
    model.options.  It will be removed in a future release.}
    
  \item{shrinkage.parameter.prior}{An object of class
    \code{\link[Boom]{GammaPrior}} describing the shrinkage parameter, a
    (see details below).}

  \item{shrinkage.parameter.prior.DEPRECATED}{This option should be set using
    model.options.  It will be removed in a future release.}

  \item{sigma.max}{ The largest supported value of each
    \code{sigma[i]}.  Truncating the support of sigma can keep
    ill-conditioned models from crashing.  This must be a positive
    number (\code{Inf} is okay), or \code{NULL}.  A \code{NULL} value
    will set \code{sigma.max = sd(y)}, which is a substantially larger
    value than one would expect, so in well behaved models this
    constraint will not affect the analysis.  }

  \item{sigma.max.DEPRECATED}{ This option should be set using
    model.options.  It will be removed in a future release.}
  
  \item{contrasts}{An optional list. See the \code{contrasts.arg} of
    \code{model.matrix.default}.  This argument is only used if a model
    formula is specified.  It can usually be ignored even then.}

  \item{na.action}{What to do about missing values.  The default is to
    allow missing responses, but no missing predictors.  Set this to
    na.omit or na.exclude if you want to omit missing responses
    altogether.}

  \item{sdy}{The standard deviation of the response variable.  This is
    used to scale default priors and \code{sigma.max} if other arguments
    are left \code{NULL}.  If all other arguments are non-\code{NULL}
    then \code{sdy} is not used.}

  \item{lags}{The number of lags in the autoregressive process for the
    coefficients.}

  \item{sigma.prior}{Either an object of class \code{\link{SdPrior}} or
    a list of such objects.  If a single \code{\link{SdPrior}} is given
    then it specifies the prior on the innovation variance for all the
    coefficients.  If a list of \code{\link{SdPrior}} objects is given,
    then each element gives the prior distribution for the corresponding
    regression coefficient.  The length of such a list must match the
    number of predictors in the dynamic regression part of the model.  }

}


\value{ Returns a list with the elements necessary to specify a dynamic
  regression model.}

\details{
  For the standard "random walk" coefficient model, the model is

  \deqn{\beta_{i, t+1} = beta_{i, t} + \epsilon_t \qquad
    \epsilon_t \sim \mathcal{N}(0, \sigma^2_i / variance_{xi})}{%
    beta[i, t+1] ~ N(beta[i, t], sigsq[i] / variance_x[i])
  }

  \deqn{ \frac{1}{\sigma^2_i} \sim Ga(a, b)}{%
    1.0 / sigsq[i] ~ Gamma(a, b)
  }

  \deqn{\sqrt{b/a} \sim sigma.mean.prior}{%
    sqrt(b / a) ~ sigma.mean.prior
  }

  \deqn{ a \sim shrinkage.parameter.prior}{%
    a ~ shrinkage.parameter.prior
  }

  That is, each coefficient evolves independently, with its own variance
  term which is scaled by the variance of the i'th column of X.  The
  parameters of the hyperprior are interpretable as: sqrt(b/a) typical
  amount that a coefficient might change in a single time period, and
  'a' is the 'sample size' or 'shrinkage parameter' measuring the degree
  of similarity in sigma[i] among the arms.

  In most cases we hope b/a is small, so that sigma[i]'s will be
  small and the series will be forecastable.  We also hope that 'a'
  is large because it means that the sigma[i]'s will be similar to
  one another.

  The default prior distribution is a pair of independent Gamma
  priors for sqrt(b/a) and a.  The mean of sigma[i] is set to .01 *
  sd(y) with shape parameter equal to 1.  The mean of the shrinkage
  parameter is set to 10, but with shape parameter equal to 1.

  If the coefficients have AR dynamics, then the model is that each
  coefficient independently follows an AR(p) process, where p is given
  by the \code{lags} argument.  Independent priors are assumed for each
  coefficient's model, with a uniform prior on AR coefficients (with
  support restricted to the finite region where the process is
  stationary), while the \code{sigma.prior} argument gives the prior for
  each coefficient's innovation variance.

}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott
}

\examples{
## Setting the seed to avoid small sample effects resulting from small
## number of iterations.
set.seed(8675309)
n <- 1000
x <- matrix(rnorm(n))

# beta follows a random walk with sd = .1 starting at -12.
beta <- cumsum(rnorm(n, 0, .1)) - 12

# level is a local level model with sd = 1 starting at 18.
level <- cumsum(rnorm(n)) + 18

# sigma.obs is .1
error <- rnorm(n, 0, .1)

y <- level + x * beta + error
par(mfrow = c(1, 3))
plot(y, main = "Raw Data")
plot(x, y - level, main = "True Regression Effect")
plot(y - x * beta, main = "Local Level Effect")

ss <- list()
ss <- AddLocalLevel(ss, y)
ss <- AddDynamicRegression(ss, y ~ x)
## In a real appliction you'd probably want more than 100
## iterations. See comment above about the random seed.
model <- bsts(y, state.specification = ss, niter = 100, seed = 8675309)
plot(model, "dynamic", burn = 10)
}

\seealso{
  \code{\link{bsts}}.
  \code{\link[Boom]{SdPrior}}
  \code{\link[Boom]{NormalPrior}}
}

\keyword{models}
