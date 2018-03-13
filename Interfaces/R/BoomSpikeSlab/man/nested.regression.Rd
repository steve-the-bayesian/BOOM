\name{nested.regression}

\Rdversion{1.1}
\alias{NestedRegression}

\title{
  Nested Regression
}

\description{
 Fits a Bayesian hierarchical regression model to data nested within groups.
 The model is

 \deqn{%
          y_{ig}  \sim N(x_i  \beta_g, \sigma^2) \\ %
     1 / \sigma^2 \sim Gamma(df/2, ss/2)        \\ %
          \beta_g \sim N(b, V)                  \\ %
        }{                                        %
          y[i,g] ~ N(x[i,] * beta[, g], sigma^2)  %
     1 / sigma^2 ~ Gamma(df/2, ss/2)              %
        beta[,g] ~ N(b, V)                        %
      }

  Optional hyperprior distributions can be supplied to the prior parameters.
  \deqn{%
                        b ~ N(prior.mean, prior.variance)       \\ %
                        V ~ InverseWishart(df, variance.guess). \\ %
                }{                                               %
                        b ~ N(prior.mean, prior.variance)        %
                        V ~ InverseWishart(df, variance.guess).  %
              }

  Either hyperprior can be omitted, in which case the corresponding
  prior parameter is assumed fixed at the user-supplied value.
}

\usage{
NestedRegression(response,
                 predictors,
                 group.id,
                 residual.precision.prior = NULL,
                 coefficient.prior = NULL,
                 coefficient.mean.hyperprior = NULL,
                 coefficient.variance.hyperprior = NULL,
                 suf = NULL,
                 niter,
                 ping = niter / 10,
                 sampling.method = c("ASIS", "DA"),
                 seed = NULL)

}

\arguments{

  \item{response}{A numeric vector.  The response variable to be modeled.}

  \item{predictors}{A numeric matrix of predictor variables, including
    an intercept term if one is desired.  The number of rows must match
    length(response).}

  \item{group.id}{A factor (or object that can be converted using
    \code{\link{as.factor}}) naming the group to which each entry in
    \code{response} belongs.}

  \item{residual.precision.prior}{An object of type
    \code{\link[Boom]{SdPrior}} describing the prior
    distribution of the residual standard deviation.}

  \item{coefficient.prior}{An object of class MvnPrior, or \code{NULL}.
    If non-\code{NULL} this gives the initial values of the prior
    distribution of the regression coefficients in the nested regression
    model.  This argument must be non-\code{NULL} if either
    \code{coefficient.mean.hyperprior} or
    \code{coefficient.variance.hyperprior} is \code{NULL}.}

  \item{coefficient.mean.hyperprior}{An object of class
    \code{\link[Boom]{MvnPrior}}, specifying the hyperprior
    distribution for the mean of \code{coefficient.prior}.  This
    argument can also be \code{NULL}, or \code{FALSE}.  If \code{NULL}
    then a default prior will be used when learning the mean of the
    prior distribution.  If \code{FALSE} then the mean of the prior
    distribution will not be learned; the mean of the
    \code{coefficient.prior} distribution will be assumed instead.}

  \item{coefficient.variance.hyperprior}{An object of class
    \code{\link[Boom]{InverseWishartPrior}}, specifying the
    hyperprior distribution for the variance of
    \code{coefficient.prior}.  This argument can also be \code{NULL}, or
    \code{FALSE}.  If \code{NULL} then a default prior will be used when
    learning the variance of the prior distribution.  If \code{FALSE}
    then the variance of the prior distribution will not be learned; the
    variance of the \code{coefficient.prior} distribution will be
    assumed instead.}

  \item{suf}{A list, where each entry is of type
    \code{\link[Boom]{RegressionSuf}}, giving the sufficient
    statistics for each group, or \code{NULL}.  If \code{NULL}, then
    \code{suf} will be computed from \code{response}, \code{predictors},
    and \code{group.id}.  If non-\code{NULL} then these arguments will
    not be accessed, in which case they can be left unspecified.  In
    'big data' problems this can be a significant computational
    savings.}

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{The frequency with which to print status updates.}

  \item{sampling.method}{The MCMC sampling scheme that should be used.
    If either hyperprior is set to \code{FALSE} then the "DA" method
    will be used.}

  \item{seed}{The integer-valued seed (or \code{NULL}) to use for the
    C++ random number generator.}
}

\value{
  A list containing MCMC draws from the posterior distribution of model
  parameters.  Each of the following is a vector, matrix, or array, with
  first index corresponding to MCMC draws, and later indices to distinct
  parameters.
  \itemize{

    \item{ coefficients: regression coefficients.}

    \item{residual.sd: the residual standard deviation from the
      regression model.}

    \item{prior.mean: The posterior distribution of the coefficient
      means across groups.}

    \item{prior.variance: The posterior distribution of the variance
      matrix describing the distribution of regression coefficients
      across groups.  }

    \item{priors: A list of the prior distributions used to fit the
      model.}

  }
}

\details{

  Note: ASIS (Yu and Meng, 2011) has slightly better MCMC convergence,
  but is slightly slower than the classic DA (data augmentation) method,
  which alternates between sampling group-level regression coefficients
  and prior parameters.  Both methods are pretty fast.

}

\author{
  Steven L. Scott
}

\examples{
SimulateNestedRegressionData <- function() {
  beta.hyperprior.mean <- c(8, 6, 7, 5)
  xdim <- length(beta.hyperprior.mean)
  beta.hyperprior.variance <-
    rWishart(2 * xdim, diag(rep(1, xdim)), inverse = TRUE)

  number.of.groups <- 27
  nobs.per.group = 23
  beta <- rmvn(number.of.groups,
               beta.hyperprior.mean,
               beta.hyperprior.variance)

  residual.sd <- 2.4
  X <- cbind(1, matrix(rnorm(number.of.groups * (xdim - 1) * nobs.per.group),
                       ncol = xdim - 1))
  group.id <- rep(1:number.of.groups, len = nrow(X))
  y.hat <- numeric(nrow(X))
  for (i in 1:nrow(X)) {
    y.hat[i] = sum(X[i, ] * beta[group.id[i], ])
  }
  y <- rnorm(length(y.hat), y.hat, residual.sd)
  suf <- BoomSpikeSlab:::.RegressionSufList(X, y, group.id)

  return(list(beta.hyperprior.mean = beta.hyperprior.mean,
              beta.hyperprior.variance = beta.hyperprior.variance,
              beta = beta,
              residual.sd = residual.sd,
              X = X,
              y = y,
              group.id = group.id,
              suf = suf))
}

d <- SimulateNestedRegressionData()
model <- NestedRegression(suf = d$suf, niter = 500)

}
