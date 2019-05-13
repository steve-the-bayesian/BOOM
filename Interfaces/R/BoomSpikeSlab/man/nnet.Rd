% Copyright 2018 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{nnet}
\Rdversion{1.1}
\alias{BayesNnet}
\alias{HiddenLayer}
\title{
  Bayesian Feed Forward Neural Networks
}
\description{
  Fit a feed forward neural network using MCMC.
}
\usage{
BayesNnet(formula,
          hidden.layers,
          niter,
          data,
          subset,
          prior = NULL,
          expected.model.size = Inf,
          drop.unused.levels = TRUE,
          contrasts = NULL,
          ping = niter / 10,
          seed = NULL) 

HiddenLayer(number.of.nodes, prior = NULL, expected.model.size = Inf)

}

\arguments{

  \item{formula}{ A formula describing the model to be fit.  The formula
    should be additive.  The network will figure out any interactions or
    nonlinearities.
  }

  \item{hidden.layers}{
    A list of objects created by \code{\link{HiddenLayer}} defining the
    network structure.  The input layer is determined by the
    \code{formula} argument.  The terminal layer is a linear regression
    on the outputs of the final hidden layer.
  }
  
  \item{niter}{
    The number of MCMC iterations to run.  Be sure to include enough so
    you can throw away a burn-in set.
  }
  
  \item{data}{
    An optional data frame, list or environment (or object coercible by
    'as.data.frame' to a data frame) containing the variables in the
    model.  If not found in 'data', the variables are taken from
    'environment(formula)', typically the environment from which
    \code{BayesNnet} is called.
  }
  
  \item{subset}{
    an optional vector specifying a subset of observations to be used in
    the fitting process.
  }

  \item{prior}{When passed to \code{BayesNnet} this is the prior
    distribution for the terminal layer, which must be an object of
    class \code{\link{SpikeSlabPrior}},
    \code{\link{SpikeSlabPriorDirect}}, or \code{NULL}.  If \code{NULL}
    then a default prior will be used.

    When passed to \code{HiddenLayer} this is the prior distribution for
    the coefficients to that layer.  The prior is specified for a single
    output node, and the same prior is used for all nodes.  You can
    think of each hidden layer output node as a logistic regression
    model where the predictors are the outputs of the previous layer.
    This must be an object of class \code{\link{MvnPrior}},
    \code{\link{SpikeSlabGlmPrior}}, or
    \code{\link{SpikeSlabGlmPriorDirect}}.
  }

  \item{expected.model.size}{
    When \code{prior} is not specified a default spike-and-slab prior
    will be used.  The \code{expected.model.size} argument to
    \code{\link{BayesNnet}} is passed to
    \code{\link{SpikeSlabPriorDirect}}.  In \code{HiddenLayer} the
    argument is passed to \code{\link{SpikeSlabGlmPriorDirect}}.

    The parameter is used to set the prior inclusion probabilities for
    the coefficients.  If \code{p} coefficients are available then the
    prior inclusion probabilities are each set to
    \code{expected.model.size / p}.  If this ratio exceeds 1 then model
    selection is turned off and all coefficients are included.
  }

  \item{drop.unused.levels}{Logical indicating whether unobserved factor
    levels should be dropped when forming the model matrix. } 
  
  \item{contrasts}{ An optional list. See the \code{contrasts.arg}
    argument of \code{\link{model.matrix.default}}.  }

  \item{ping}{The frequency with which to print status update messages
    to the screen.  For example, if \code{ping == 10} then an update
    will be printed every 10 MCMC iterations.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

  \item{number.of.nodes}{The number of nodes in this hidden layer.  This
    must be a positive scalar integer.}

}

\details{

  The model is a feedforward neural network regression.  The model is
  fit using an MCMC algorithm based on data augmentation.  Each hidden
  node is randomly assigned a 0/1 value from its full conditional
  distribution.  Then conditional on the imputed data an MCMC draw is
  done on each latent logistic regression and on the regression model
  defining the terminal node.
  
}

\value{
  The returned object is a list with class \code{BayesNnet}.  It
  contains the following objects

  \itemize{

    \item \code{residual.sd} The standard deviation of the residuals
    from the model.

    \item \code{hidden.layer.coefficients} A list, with one element per
    hidden layer, giving the posterior draws of the hidden layer
    coefficients for that layer.  Each list element is a 3-way array
    with dimensions corresponding to
    \enumerate{
      \item MCMC iteration
      \item Input node.  For the first hidden layer each 'input node' is
      a predictor variable.
      \item Output node.  
    }
    You can think of hidden.layer.coefficients[[i]][, , j] as the
    posterior distribution of the logistic regression model defining
    node 'j' in hidden layer 'i'.

    \item \code{terminal.layer.coefficients} A matrix containing the
    MCMC draws of the model coefficients for the terminal layer.

    \item Other list elements needed to implement various methods
    (predict, plot, etc.).
  }
}

\references{
  ??
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{plot.BayesNnet}},
  \code{\link{predict.BayesNnet}}.
}

\examples{

if (require(mlbench)) {
  data(BostonHousing)
  hidden.layers <- list(
    HiddenLayer(10, expected.model.size = Inf))

  ## In real life you'd want more 50 MCMC draws.
  model <- BayesNnet(medv ~ .,
    hidden.layers = hidden.layers,
    niter = 50,
    data = BostonHousing)

  par(mfrow = c(1, 2))
  plot(model)  # plots predicted vs actual.
  plot(model, "residual") # plots 
  par(mfrow = c(1,1))
  plot(model, "structure")
  ## Examine all partial dependence plots.
  plot(model, "partial", pch = ".")

  ## Examine a single partial dependence plot.
  par(mfrow = c(1,1))
  plot(model, "lstat", pch = ".")

  ## Check out the mixing performance.
  PlotManyTs(model$terminal.layer.coefficients)
  PlotMacf(model$terminal.layer.coefficients)

  ## Get the posterior distribution of the function values for the
  ## training data.
  pred <- predict(model)

  ## Get predictions for data at new points (though in this example I'm
  ## reusing old points.
  pred2 <- predict(model, newdata = BostonHousing[1:12, ])

} else {
  cat("The Boston housing data from 'mlbench' is needed for this example.")
}

}

\keyword{models}
