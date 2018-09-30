% Copyright 2018 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{plot.BayesNnet}
\Rdversion{1.1}
\alias{plot.BayesNnet}
\alias{PlotNetworkStructure}
\alias{PlotBayesNnetPredictions}
\alias{PlotBayesNnetResiduals}

\title{
  Plot a Bayesian Neural Network
}

\description{
  The default plot is a barplot of the marginal inclusion probabilities
  for each variable, as obtained by
  \code{\link{PlotMarginalInclusionProbabilities}}.  Other interesting
  plots can be obtained by supplying a string as the second argument.

}

\usage{
  \method{plot}{BayesNnet}(x,
     y = c("predicted", "residual", "structure", "partial", "help"),
     ...)

  PlotBayesNnetPredictions(model, burn = SuggestBurn(model), ...)

  PlotBayesNnetResiduals(model, burn = SuggestBurn(model), ...)

  PlotNetworkStructure(model, ...)

}

\arguments{
  \item{model}{ An object of class \code{BayesNnet}. }
  
  \item{x}{
    An object of class \code{BayesNnet}.  The name \code{x} is required
    to conform with the \code{plot} generic function signature.
  }
  
  \item{y}{
    The type of plot desired, or the name of the variable to plot
    against.  The name \code{y} is required to conform with the
    \code{plot} generic function signature.

    If \code{y} matches (or partially matches) one of the names in the
    function signature, then the corresponding plot function handles the
    plot request.

    If \code{y} is the string "help" then this help page is opened in a
    browser.

    If \code{y} matches (or partially matches) the name of one of the
    variables in the training data, then a partial dependence plot vs
    that variable is produced.
  }

  \item{burn}{
    The number of MCMC iterations to discard as burn-in.
  }

  \item{\dots}{ Additional arguments passed to the specific functions
    that do the plotting.  }

}

\details{ The actual plotting will be handled by
  \code{\link{PlotMarginalInclusionProbabilities}},
  \code{\link{PlotLmSpikeCoefficients}},
  \code{\link{PlotLmSpikeResiduals}}, or \code{\link{PlotModelSize}}.
  See the appropriate function for more options.}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{BayesNnet}}
  \code{\link{PartialDependencePlot}}
}

\examples{
}

\keyword{models}
\keyword{regression}
