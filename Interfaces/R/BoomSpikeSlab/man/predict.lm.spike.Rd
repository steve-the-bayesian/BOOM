% Copyright 2010 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{predict.lm.spike}
\Rdversion{1.1}
\alias{predict.lm.spike}
\alias{predict.logit.spike}
\alias{predict.probit.spike}
\alias{predict.poisson.spike}
\alias{predict.qreg.spike}
\alias{predict.BayesNnet}
\title{
  Predictions using spike-and-slab regression.
}
\description{
  Generate draws from the posterior predictive distribution of a spike
  and slab regression.
}

\usage{
\method{predict}{lm.spike}(object, newdata = NULL, burn = 0,
    na.action = na.pass, mean.only = FALSE, ...)

\method{predict}{logit.spike}(object, newdata, burn = 0,
    type = c("prob", "logit", "link", "response"),
    na.action = na.pass, ...)

\method{predict}{poisson.spike}(object, newdata = NULL,
    exposure = NULL, burn = 0,
    type = c("mean", "log", "link", "response"),
    na.action = na.pass, ...)

\method{predict}{probit.spike}(object, newdata, burn = 0,
    type = c("prob", "probit", "link", "response"),
    na.action = na.pass, ...)

\method{predict}{qreg.spike}(object, newdata, burn = 0,
    na.action = na.pass, ...)

\method{predict}{BayesNnet}(object, newdata = NULL, burn = 0,
    na.action = na.pass, mean.only = FALSE, seed = NULL, ...)

}

\arguments{
  \item{object}{
    A model object of class \code{lm.spike}, \code{logit.spike}, etc.
  }

  \item{newdata}{ Either \code{NULL}, or else a data frame, matrix, or
    vector containing the predictors needed to make the prediction.

    If \code{newdata} is \code{NULL} then the predictors are taken from
    the training data used to create the model object.  Note that
    \code{object} does not store its training data, so the data objects
    used to fit the model must be present for the training data to be
    recreated.

    If \code{newdata} is a \code{data.frame} it must contain variables
    with the same names as the data frame used to fit \code{object}.  If
    it is a \code{matrix}, it must have the same number of columns as
    \code{object$beta}.  An intercept term will be implicitly added if
    the number of columns is too small by one.  If the dimension of
    \code{object$beta} is 1 or 2, then \code{newdata} can be a vector. }

  \item{exposure}{ A vector of positive real numbers the same size as
    newdata, or \code{NULL}.  If both \code{newdata} and \code{exposure}
    are \code{NULL} then \code{exposure} is taken to be the exposure
    from the training data.  If \code{newdata} is supplied and
    \code{exposure} is \code{NULL} then \code{exposure} is taken to be 1
    for all observations.  }
  
  \item{burn}{
    The number of MCMC iterations in the object to be discarded as
    burn-in.
  }

  \item{na.action}{a function which indicates what should happen when
    the data contain \code{NA}'s.  The default is set by the
    \code{na.action} setting of \code{\link{options}}, and is
    \code{\link{na.fail}} if that is unset.  The "factory-fresh" default
    is \code{\link{na.omit}}.  Another possible value is \code{NULL}, no
    action.  Value \code{\link{na.exclude}} can be useful.}

  \item{type}{ The type of prediction desired.

    For \code{logit.spike}, \code{prob} means the prediction is returned
    on the probability scale, while \code{logit} returns the scale of
    the linear predictor.  Probits work similarly to logits.

    For \code{poisson.spike}, \code{mean} means the prediction is
    returned on the scale of the data, while \code{log} means it is on
    the scale of the linear predictor.

    Both cases also accept \code{link} and \code{response} for
    compatibility with \code{predict.glm}.  }

  \item{mean.only}{Logical.  If \code{TRUE} then return the posterior
    mean of the predictive distribution.  If \code{FALSE} then return
    the entire distribution.}

  \item{seed}{Random seed for the C++ random number generator.  This is
    only needed for models that require C++ to implement their predict
    method.}
  
  \item{\dots}{Unused, but present for compatibility with generic
    \code{predict}.}

}
\value{
  Returns a matrix of predictions, with each row corresponding to a row
  in newdata, and each column to an MCMC iteration.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{summary.lm.spike}}
  \code{\link{plot.lm.spike}}
}

\examples{
  niter <- 1000
  n <- 100
  p <- 10
  ngood <- 3

  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- rep(0, p)
  good <- sample(1:p, ngood)
  beta[good] <- rnorm(ngood)
  sigma <- 1

  y <- rnorm(n, x \%*\% beta, sigma)
  model <- lm.spike(y ~ x - 1, niter=niter)
  plot(model)
  plot.ts(model$beta)
  hist(model$sigma)  ## should be near true value

  new.x <- cbind(1, matrix(rnorm(100 * (p-1)), ncol = (p-1)))
  pred <- predict(model, newdata = new.x, burn = 100)
}
