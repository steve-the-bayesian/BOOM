# Copyright 2021 Steven L. Scott. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

## Objects defined here must have C++ analogs in
## <r_interface/create_mixture_components.?pp>
## All objects must have entries named 'data', 'data.type', and 'name'

CheckMixtureComponent <- function(..., fun = "", formula = "", data = "") {
  ## This is a reimplementation of 'stopifnot' with better error
  ## messages.  It will actually tell you which mixture component
  ## caused the error.
  ##
  ## Args:
  ##   ...: The collection of condtions that must be satisfied for the
  ##     mixture component to be valid.
  ##   fun: The name of the function doing the check.  This is
  ##     intended to be the constructor function for the mixture
  ##     component.
  ##   formula: If the mixture component contains a formula
  ##     (e.g. regression and logistic regression) it can be specified
  ##     to better identify the component that had the problem.
  ##   data:  The name of the 'data' argument passed to 'fun'.
  ##
  ## Effects:
  ##   If the conditions specified in ... are satisfied then nothing
  ##   happens.  Otherwise stop() is called with an error message
  ##   explaining which condition failed, and giving any information
  ##   provided by fun, formula, and data.
  conditions <- list(...)
  n <- length(conditions)
  mc <- match.call()
  for (i in 1:n) {
    if (!(is.logical(r <- conditions[[i]]) &&
                          !any(is.na(r)) &&
                          all(r))) {
      ch <- deparse(mc[[i + 1]], width.cutoff = 60L)
        if (length(ch) > 1L)
            ch <- paste(ch[1L], "....")
      msg <- paste(ch, " is not ", if (length(r) > 1L) "all ", "TRUE", sep = "")
      if (fun != "") {
        msg <- paste(msg, " In call to ", fun, sep = "")
      }
      if (formula != "") {
        msg <- paste(msg, " with formula = ", formula, sep = "")
      }
      if (data != "") {
        msg <- paste(msg, " with data = ", data, sep = "")
      }
      stop(msg, call. = FALSE)
    }
  }
}

###=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
NormalMixtureComponent <- function(data, prior = NULL, group.id = NULL,
                                   name = "", ...) {
  ## Args:
  ##   data: A numeric vector to be modeled as a mixture of normals.
  ##   prior:  An object of class NormalInverseGammaPrior.
  ##   group.id: an optional factor indicating group membership.
  ##   name: A character string naming this mixture component.  This
  ##     is especially important if multiple mixture components are
  ##     present in an ensemble.
  ##
  ## Returns:
  ##   An object of class NormalMixtureComponent

  data.name <- deparse(substitute(data))
  fun <- "NormalMixtureComponent"
  CheckMixtureComponent(is.numeric(data), fun = fun, data = data.name)
  CheckMixtureComponent(length(data) > 0, fun = fun, data = data.name)

  if (is.null(group.id)) {
    group.id <- rep(1, length(data))
  }
  data <- split(data, group.id)

  if (is.null(prior)) {
    y <- unlist(data)
    ybar <- mean(y, na.rm = TRUE)
    sample.sd <- sd(y, na.rm = TRUE)
    prior <- NormalInverseGammaPrior(mu.guess = ybar,
                                     sigma.guess = sample.sd,
                                     ...)
  }
  CheckMixtureComponent(inherits(prior, "NormalInverseGammaPrior"),
             fun = fun, data = data.name)

  model <- list(data = data,
                prior = prior,
                data.type = "double.data",
                name = name)
  class(model) <- c("NormalMixtureComponent", "MixtureComponent")
  return(model)
}

###=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
PoissonMixtureComponent <-
  function(data, prior = NULL, group.id = NULL, name = "",
           prior.mean = mean(data), prior.sample.size = 1, ...) {
  ## Args:
  ##   data:  A numeric vector of counts to be modeled by a mixture of Poissons.
  ##   prior: An object of class GammaPrior, where the first (shape,
  ##     or 'alpha') parameter corresponds to a sum of prior events,
  ##     and the second (scale, or 'beta') parameter corresponds to a
  ##     prior sample size.
  ##   group.id: an optional factor indicating group membership.
  ##   name: A character string naming this mixture component.  This
  ##     is especially important if multiple mixture components are
  ##     present in an ensemble.
  ##   prior.mean: The prior mean of the Poisson rate parameter.  This
  ##     argument is only used if 'prior' is omitted.
  ##   prior.sample.size: The number of observations worth of weight
  ##     assigned to 'prior.mean'.  Used only if 'prior' is omitted.
  ##   ...: Extra arguments passed to GammaPrior (if 'prior' is
  ##     omitted)
  ## Returns:
  ##   An object of class PoissonMixtureComponent.
  fun <- "PoissonMixtureComponent"
  data.name <- deparse(substitute(data))
  CheckMixtureComponent(is.numeric(data), fun = fun, data = data.name)
  CheckMixtureComponent(all(is.na(data) | data >= 0),
                        fun = fun, data = data.name)
  if (is.null(group.id)) {
    group.id <- rep(1, length(data))
  }
  mixture.data <- split(data, group.id)

  if (is.null(prior)) {
    prior <- GammaPrior(a = prior.mean * prior.sample.size,
                        b = prior.sample.size,
                        ...)
  }
  ans <- list(data = mixture.data,
              prior = prior,
              data.type = "int.data",
              name = name)
  class(ans) <- c("PoissonMixtureComponent", "MixtureComponent")
  return(ans)
}
###=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
ZeroInflatedPoissonMixtureComponent <- function(
    data, lambda.prior = NULL, zero.probability.prior = NULL,
    group.id = NULL, name = "") {
  ## Args:
  ##   data:  A numeric vector of counts to be modeled by a mixture of Poissons.
  ##   lambda.prior: An object of class GammaPrior, where the first
  ##     (shape, or 'alpha') parameter corresponds to a sum of prior
  ##     events, and the second (scale, or 'beta') parameter
  ##     corresponds to a prior sample size.
  ##   zero.probability.prior: An object of class BetaPrior to use as
  ##     the prior for the probability of a forced zero.
  ##   group.id: an optional factor indicating group membership.
  ##   name: A character string naming this mixture component.  This
  ##     is especially important if multiple mixture components are
  ##     present in an ensemble.
  ## Returns:
  ##   An object of class ZeroInflatedPoissonMixtureComponent.
  fun <- "ZeroInflatedPoissonMixtureComponent"
  data.name <- deparse(substitute(data))
  CheckMixtureComponent(is.numeric(data), fun = fun, data = data.name)
  CheckMixtureComponent(all(is.na(data) | data >= 0), fun = fun,
                        data = data.name)
  if (is.null(group.id)) {
    group.id <- rep(1, length(data))
  }
  ybar <- mean(data)
  mixture.data <- split(data, group.id)

  if (is.null(zero.probability.prior)) {
    ## Default prior for zero probabilities is uniform.
    zero.probability.prior <- BetaPrior(1, 1)
  }
  CheckMixtureComponent(inherits(zero.probability.prior, "BetaPrior"),
                        fun = fun, data = data.name)

  if (is.null(lambda.prior)) {
    ## Default prior is a single observation with the average y, note
    ## that this might be a bit too small if many, observations
    ## are zero.
    lambda.prior <- GammaPrior(ybar, 1)
  }
  CheckMixtureComponent(inherits(lambda.prior, "GammaPrior"),
             fun = fun, data = data.name)

  ans <- list(data = mixture.data,
              gamma.prior = lambda.prior,
              beta.prior = zero.probability.prior,
              data.type = "int.data",
              name = name)
  class(ans) <- c("ZeroInflatedPoissonMixtureComponent", "MixtureComponent")
  return(ans)
}
###=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
ZeroInflatedLognormalMixtureComponent <-
  function(data, beta.prior = NULL, normal.inverse.gamma.prior = NULL,
           group.id = NULL, name = "", ...)
{
  ## Args:
  ##   data: A numeric vector of non-negative data to be modeled as a
  ##     mixture of zero-inflated log normal models.
  ##   beta.prior: An object of class BetaPrior to use as the prior
  ##     for the probability of positive data.
  ##   normal.inverse.gamma.prior: An object of class
  ##     NormalInverseGammaPrior to be used as a prior for the normal
  ##     model for the log of the positive data.
  ##   group.id: an optional factor indicating group membership.
  ##   name: A character string naming this mixture component.  This
  ##     is especially important if multiple mixture components are
  ##     present in an ensemble.
  ##   ...: Extra arguments passed to SdPrior, for the portion
  ##     of the prior relating to the standard deviation of log y.
  ## Returns:
  ##   An object of class ZeroInflatedLognormalMixtureComponent that
  ##   is a list containing the preceding arguments, after potential
  ##   coercion and type checking.
  fun <- "ZeroInflatedLognormalMixtureComponent"
  data.name <- deparse(substitute(data))
  CheckMixtureComponent(is.numeric(data), fun = fun, data = data.name)
  CheckMixtureComponent(length(data) > 0, fun = fun, data = data.name)
  CheckMixtureComponent(all(is.na(data) | data >= 0), fun = fun,
                        data = data.name)
  CheckMixtureComponent(any(data > 0), fun = fun, data = data.name)

  if (is.null(beta.prior)) {
    beta.prior <- BetaPrior(1, 1)
  }
  CheckMixtureComponent(inherits(beta.prior, "BetaPrior"), fun = fun,
                        data = data.name)

  if (is.null(normal.inverse.gamma.prior)) {
    logy <- log(data[data > 0])
    logy <- logy[!is.na(logy)]
    normal.inverse.gamma.prior <-
      NormalInverseGammaPrior(mean(logy), 1.0, sd(logy), 1.0, ...)
  }
  CheckMixtureComponent(inherits(normal.inverse.gamma.prior,
                                 "NormalInverseGammaPrior"),
                        fun = fun, data = data.name)

  if (is.null(group.id)) {
    data <- list(data)
  } else {
    data <- split(data, group.id)
  }

  ans <- list(data = data,
              beta.prior = beta.prior,
              normal.inverse.gamma.prior = normal.inverse.gamma.prior,
              data.type = "double.data",
              name = name)
  class(ans) <- c("ZeroInflatedLognormalMixtureComponent",
                  "MixtureComponent")
  return(ans)
}

###=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
RegressionMixtureComponent <- function(formula, data, prior = NULL,
                                       contrasts = NULL,
                                       drop.unused.levels = TRUE,
                                       group.id = NULL,
                                       name = "", ...) {
  ## I want to support RegressionMixtureComponent(y ~ x, data =
  ## data.frame, subject.id = variable.in.data.frame)

  fun <- "RegressionMixtureComponent"
  formula.name <- deparse(formula)
  data.name <- deparse(substitute(data))

  ##------ Begin stuff copied from lm (by way of spikeslab package) -----
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- drop.unused.levels
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  y <- model.response(mf, "numeric")

  x <- model.matrix(mt, mf, contrasts)
  if (missing(prior)) {
    prior <- SpikeSlabPrior(x, y, ...)
  }
  ##------ End stuff copied from lm (by way of spikeslab package) -----

  if (is.null(group.id)) {
    group.id <- rep(1, length(y))
  }

  ## Split data into groups with the same group.id.
  y.list <- split(y, group.id)
  x.list <- split.data.frame(x, group.id)
  mixture.data = list()
  unique.groups <- unique(group.id)
  for (i in seq(along = unique.groups)) {
    # The lower-case names 'x' and 'y' are assumed by the
    # corresponding c++ code.
    mixture.data[[i]] <- list(y = y.list[[i]], x = x.list[[i]])
  }

  CheckMixtureComponent(is.null(prior) || inherits(prior, "SpikeSlabPrior"),
             fun = fun, formula = formula.name, data = data.name)

  ans <- list(data = mixture.data,
              prior = prior,
              data.type = "regression.data",
              contrasts = attr(x, "contrasts"),
              xlevels = .getXlevels(mt, mf),
              call = cl,
              terms = mt,
              sample.sd = sd(y, na.rm = TRUE),
              name = name
              )

  class(ans) <- c("RegressionMixtureComponent", "MixtureComponent")
  return(ans)
}

###=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
LogitMixtureComponent <-
  function(formula, data, prior = NULL, contrasts = NULL,
           drop.unused.levels = TRUE, group.id = NULL, name = "", ...)
{
  fun <- "LogitMixtureComponent"
  formula.name <- deparse(formula)
  data.name <- deparse(substitute(data))

  ##------------ Begin stuff copied from glm (by way of spikeslab package ----
  cl <- match.call()
  mf <- match.call(expand.dots = FALSE)
  m <- match(c("formula", "data", "na.action"), names(mf), 0L)
  mf <- mf[c(1L, m)]
  mf$drop.unused.levels <- drop.unused.levels
  mf[[1L]] <- as.name("model.frame")
  mf <- eval(mf, parent.frame())
  mt <- attr(mf, "terms")
  y <- model.response(mf, "any")

  if (!is.null(dim(y)) && length(dim(y)) > 1) {
    CheckMixtureComponent(length(dim(y)) == 2, ncol(y) == 2, fun = fun,
               formula = formula.name, data = data.name)
    ## If the user passed a formula like "cbind(successes, failures) ~
    ## x", then y will be a two column matrix
    ny <- as.integer(y[, 1] + y[, 2])
    y <- as.integer(y[, 1])
  } else {
    ## The following line admits y's which are TRUE/FALSE, 1/0 or 1/-1.
    y <- as.integer(y > 0)
    ny <- as.integer(rep(1, length(y)))
  }

  x <- model.matrix(mt, mf, contrasts)
  if (missing(prior)) {
    prior <- SpikeSlabPrior(x, y, ...)
  }
  ##------------ End stuff copied from glm (by way of spikeslab package ----

  if (is.null(group.id)) {
    group.id <- rep(1, nrow(x))
  }
  ## Split data into groups with the same group.id
  mixture.data <- list()
  y.list <- split(y, group.id)
  ny.list <- split(ny, group.id)
  x.list <- split.data.frame(x, group.id)
  unique.groups <- unique(group.id)
  for (i in seq(along = unique.groups)) {
    mixture.data[[i]] <- list(y = y.list[[i]],
                              n = ny.list[[i]],
                              x = x.list[[i]])
  }

  ans <- list(data = mixture.data,
              prior = prior,
              data.type = "binomial.regression.data",
              contrasts = attr(x, "contrasts"),
              xlevels = .getXlevels(mt, mf),
              call = cl,
              terms = mt,
              sample.sd = sd(y, na.rm = TRUE),
              name = name)
  class(ans) <- c("LogitMixtureComponent", "MixtureComponent")
  return(ans)
}

##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
## MultinomialLogitMixtureComponent <- function(formula, data, prior,
## name = "", ...) { }

##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
MultinomialMixtureComponent <- function(data, prior = NULL,
                                        group.id = NULL, name = "", ...) {
  ## Args:
  ##   data: a factor containing the data to be modeled as a mixture
  ##     of multinomials
  ##   prior: An object of class "DirichletPrior" giving the prior
  ##     distribution for the multinomial probabilities.  The
  ##     dimension of this distribution match the number of levels in
  ##     'data'.
  ##   group.id: an optional factor indicating group membership.
  ##   name: A character string naming this mixture component.  This
  ##     is especially important if multiple mixture components are
  ##     present in an ensemble.
  ##   ...: Extra arguments passed to DirichletPrior, used only if
  ##     prior is missing.
  ## Returns:
  ##   An object of class MultinomialMixtureComponent, which is a list
  ##   containing the preceding arguments, after potential coercion
  ##   and type checking.
  fun <- "MultinomialMixtureComponent"
  data.name <- deparse(substitute(data))
  data <- as.factor(data)
  nlevels <- length(levels(data))
  data.levels <- levels(data)
  if (is.null(prior)) {
    prior <- DirichletPrior(rep(1, nlevels))
  }

  CheckMixtureComponent(inherits(prior, "DirichletPrior"),
             fun = fun, data = data.name)
  CheckMixtureComponent(length(prior$prior.counts) == nlevels,
             fun = fun, data = data.name)

  if (is.null(group.id)) {
    data <- list(data)
  } else {
    data <- split(data, group.id)
  }

  ans <- list(data = data,
              prior = prior,
              data.type = "factor.data",
              levels = data.levels,
              name = name)
  class(ans) <- c("MultinomialMixtureComponent", "MixtureComponent")
  return(ans)
}

##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
MarkovMixtureComponent <- function(data, prior = NULL,
                                   group.id = NULL, name = "", ...) {
  ## Args:
  ##   data: a factor containing the data to be modeled as a mixture
  ##     of Markov transitions.
  ##   prior: An object of class "MarkovPrior".  The dimension of this
  ##     distribution match the number of levels in 'data'.
  ##   group.id: an optional factor indicating group membership.
  ##   name: A character string naming this mixture component.  This
  ##     is especially important if multiple mixture components are
  ##     present in an ensemble.
  ##   ...: Extra arguments passed to MarkovPrior, used only if
  ##     prior is missing.
  ## Returns:
  ##   An object of class MarkovMixtureComponent, which is a list
  ##   containing the preceding arguments, after potential coercion
  ##   and type checking.
  fun <- "MarkovMixtureComponent"
  data.name <- deparse(substitute(data))
  data <- as.factor(data)
  nlevels <- length(levels(data))
  data.levels <- levels(data)
  if (is.null(prior)) {
    prior <- MarkovPrior(state.space.size = nlevels, ...)
  }
  CheckMixtureComponent(inherits(prior, "MarkovPrior"),
             fun = fun, data = data.name)
  if (is.null(group.id)) {
    data <- list(data)
  } else {
    data <- split(data, group.id)
  }

  ans <- list(data = data,
              prior = prior,
              data.type = "markov.data",
              levels = data.levels,
              name = name)
  class(ans) <- c("MarkovMixtureComponent", "MixtureComponent")
  return(ans)
}

##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
MvnMixtureComponent <- function(data, prior = NULL, group.id = NULL,
                                name = "", ...) {
  ## Args:
  ##   data: a matrix where each row is a an observation to be modeled
  ##     as a mixture of multivariate normals.
  ##   prior:  An object of class 'NormalInverseWishartPrior'
  ##   group.id: an optional factor indicating group membership.
  ##   name: A character string naming this mixture component.  This
  ##     is especially important if multiple mixture components are
  ##     present in an ensemble.
  ##   ...: Extra arguments passed to NormalInverseWishartPrior, used
  ##     only if prior is missing.
  fun <- "MvnMixtureComponent"
  data.name <- deparse(substitute(data))
  data <- as.matrix(data)

  if (is.null(prior)) {
    ybar <- colMeans(data);
    sigma <- var(data)
    prior <- NormalInverseWishartPrior(mean.guess = ybar,
                                       variance.guess = sigma,
                                       ...)
  }
  CheckMixtureComponent(inherits(prior, "NormalInverseWishartPrior"),
             fun = fun, data = data.name)

  if (is.null(group.id)) {
    group.id <- rep(1, nrow(data))
  }

  mixture.data <- split.data.frame(data, group.id)

  ans <- list(data = mixture.data,
              prior = prior,
              data.type = "vector.data",
              name = name)
  class(ans) <- c("MvnMixtureComponent", "MixtureComponent")
  return(ans)
}

##=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
IndependentMvnMixtureComponent <-
  function(data,
           prior.mean.guess = apply(data, 2, mean),
           prior.mean.sample.size = 1.0,
           prior.sd.guess = apply(data, 2, sd),
           prior.sd.sample.size = 1.0,
           sigma.upper.limit = NULL,
           group.id = NULL,
           name = "") {
    ## Args:
    ##   data: a matrix where each row is a an observation to be modeled
    ##     as a mixture of multivariate normals.
    ##   prior.mean:  A vector giving the prior mean for each variable in data.
    ##   prior.mean.sample.size: A vector giving the weight, in terms of
    ##     prior observations, to be assigned to prior.mean.
    ##   prior.sd.guess: A vector giving a guess at the standard
    ##     deviation for each variable in data.
    ##   prior.sd.sample.size: A vector giving the weight, in therms of
    ##     prior observations, to be assigned to prior.sd.guess.
    ##   sigma.upper.limit: A vector giving the maximum acceptable
    ##     value of the standard deviation for each variable in data.
    ##     If NULL then all upper limits will be assumed infinite.
    ##   group.id: an optional factor indicating group membership.
    ##   name: A character string naming this mixture component.  This
    ##     is especially important if multiple mixture components are
    ##     present in an ensemble.
    fun <- "IndependentMvnMixtureComponent"
    data.name <- deparse(substitute(data))
    data <- as.matrix(data)

    if (length(prior.mean.sample.size) == 1) {
      prior.mean.sample.size <-
        rep(prior.mean.sample.size, length(prior.mean.guess))
    }

    if(length(prior.sd.sample.size) == 1) {
      prior.sd.sample.size <-
        rep(prior.sd.sample.size, length(prior.mean.guess))
    }

    if(is.null(sigma.upper.limit)) {
      sigma.upper.limit <- rep(Inf, length(prior.mean.guess))
    }

    CheckMixtureComponent(length(prior.mean.guess) ==
                          ncol(data),
                          fun = fun,
                          data = data.name)

    CheckMixtureComponent(length(prior.mean.guess) ==
                          length(prior.mean.sample.size),
                          fun = fun,
                          data = data.name)

    CheckMixtureComponent(length(prior.mean.guess) ==
                          length(prior.sd.guess),
                          fun = fun,
                          data = data.name)

    CheckMixtureComponent(length(prior.mean.guess) ==
                          length(prior.sd.sample.size),
                          fun = fun,
                          data = data.name)

    CheckMixtureComponent(length(sigma.upper.limit) ==
                          length(prior.sd.sample.size),
                          fun = fun,
                          data = data.name)

    if (is.null(group.id)) {
      group.id <- rep(1, nrow(data))
    }

    mixture.data <- split.data.frame(data, group.id)

    ans <- list(data = mixture.data,
                prior.mean.guess = prior.mean.guess,
                prior.mean.sample.size = prior.mean.sample.size,
                prior.sd.guess = prior.sd.guess,
                prior.sd.sample.size = prior.sd.sample.size,
                sigma.upper.limit = sigma.upper.limit,
                data.type = "vector.data",
                name = name)
    class(ans) <- c("IndependentMvnMixtureComponent",
                    "MixtureComponent")
    return(ans)
  }
