\name{shrinkage.regression}

\Rdversion{1.1}
\alias{ShrinkageRegression}
\alias{CoefficientGroup}

\title{
  Shrinking Regression Coefficients
}

\description{
 Fits a Bayesian regression model with a shrinkage prior on the coefficient.
 The model is

 \deqn{%
            y_i  \sim N(x_i  \beta, \sigma^2) \\ %
     1 / \sigma^2 \sim Gamma(df/2, ss/2)        \\ %
  g_1(\beta) \sim N(b1, v1)                \\ %
  g_2(\beta) \sim N(b2, v2)                \\ %
  \dots }{ %
            y[i] ~ N(x[i,] * beta, sigma^2)  %
     1 / sigma^2 ~ Gamma(df/2, ss/2)         %
  group(beta, 1) ~ N(b1, v1)                 %
  group(beta, 2) ~ N(b2, v2)                 %
  ... }

  In this notation, \eqn{g_k(\beta) \sim N(b_k, v_k)}{group(beta, k) ~
  N(bk, vk)} indicates that the subset of coefficients in group k are a
  priori independent draws from the specified normal distribution. In
  addition, each subset-level prior may include a hyperprior, in which
  case the subset-level prior parameters will be updated as part of the
  MCMC.  The hyperprior has the form of independent priors on the mean
  and precision parameters:
  \deqn{%
                      b_i ~ N(prior.mean, prior.variance)   \\ %
                  1 / v_i ~ Chisq(df, guess.at.sd).          \\ %
                }{%
                      bi ~ N(prior.mean, prior.variance)
                1.0 / vi ~ Chisq(df, guess.at.sd).
              }

}

\usage{
ShrinkageRegression(response, predictors, coefficient.groups,
                    residual.precision.prior = NULL,
                    suf = NULL, niter, ping = niter / 10,
                    seed = NULL)

CoefficientGroup(indices, mean.hyperprior = NULL, sd.hyperprior = NULL,
                 prior = NULL)

}

\arguments{

  \item{response}{The numeric vector of responses.}

  \item{predictors}{The matrix of predictors, including an intercept
    term, if desired.}

  \item{coefficient.groups}{A list of objects of type
    \code{\link{CoefficientGroup}}, defining the pattern in which the
    coefficients should be shrunk together.  Each coefficient must
    belong to exactly one \code{CoefficientGroup}.}

  \item{residual.precision.prior}{An object of type
    \code{\link[Boom]{SdPrior}} describing the prior distribution of the
    residual standard deviation.}

  \item{suf}{An object of class \code{\link[Boom]{RegressionSuf}} containing
    the sufficient statistics for the regression model.  If this is
    \code{NULL} then it will be computed from \code{response} and
    \code{predictors}.  If it is supplied then \code{response} and
    \code{predictors} are not used and can be left missing.}

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{The frequency with which to print status updates.}

  \item{seed}{The integer-valued seed (or \code{NULL}) to use for the
    C++ random number generator.}

  \item{indices}{A vector of integers giving the positions of the regression
    coefficients that should be viewed as exchangeable.}

  \item{mean.hyperprior}{A \code{\link[Boom]{NormalPrior}} object describing
    the hyperprior distribution for the average coefficient.}

  \item{sd.hyperprior}{An \code{\link[Boom]{SdPrior}} object describing the
    hyperprior distribution for the standard deviation of the
    coefficients.}

  \item{prior}{An object of type \code{\link[Boom]{NormalPrior}} giving the
    initial value of the distribution describing the collection of
    coefficients in this group.  If either hyperprior is \code{NULL}
    then the corresponding prior parameter will not be updated.  If both
    hyperpriors are non-\code{NULL} then this parameter can be left
    unspecified.}

}

\value{

  \code{ShrinkageRegression} returns a list containing MCMC draws from
  the posterior distribution of model parameters.  Each of the following
  is a matrix, with rows corresponding to MCMC draws, and columsn to
  distinct parameters.
  \itemize{
    \item coefficients: regression coefficients.

    \item residual.sd: the residual standard deviation from the
      regression model.

    \item group.means: The posterior distribution of the mean of each
      coefficient group.  If no mean hyperprior was assigned to a
      particular group, then the value here will be a constant (the
      values supplied by the \code{prior} argument to
      \code{CoefficientGroup} for that group).

    \item group.sds: The posterior distribution of the standard
      deviation of each coefficient group.  If no sd.hyperprior was
      assigned to a particular group, then the value here will be a
      constant (the values supplied by the \code{prior} argument to
      \code{CoefficientGroup} for that group).
    }

  \code{CoefficientGroup} is a configuration utility used to define
  which coefficients should be shrunk together.  It returns an object
  (list) formatted in the manner expected by
  \code{ShrinkageRegression}.
}

\author{
  Steven L. Scott
}

\examples{
b0 <- -1
b1 <- rnorm(20, 3, .2)
b2 <- rnorm(30, -4, 7)
nobs <- 10000
beta <- c(b0, b1, b2)

X <- cbind(1, matrix(rnorm(nobs * (length(beta) - 1)), nrow = nobs, ncol = length(beta) - 1))
y.hat <- X \%*\% beta
y <- rnorm(nobs, y.hat, .5)

groups <- list(intercept = CoefficientGroup(1, prior = NormalPrior(0, 100)),
               first = CoefficientGroup(2:21,
                                        mean.hyperprior = NormalPrior(0, 100),
                                        sd.hyperprior = SdPrior(.2, 1)),
               second = CoefficientGroup(22:51,
                                         mean.hyperprior = NormalPrior(0, 100),
                                         sd.hyperprior = SdPrior(7, 1)))

model <- ShrinkageRegression(y, X, groups,
                             residual.precision.prior = SdPrior(.5, 1),
                             niter = 1000)

}
