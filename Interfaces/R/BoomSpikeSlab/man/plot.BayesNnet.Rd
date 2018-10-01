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

    \itemize{

      \item{"predicted" (the default) plot actual vs predicted values
        using \code{PlotBayesNnetPredictions}.}
      
      \item{"residual" plot residuals vs predicted values using
        \code{PlotBayesNnetResiduals}.}
      
      \item{"structure" plot network structure using
        \code{PlotNetworkStructure}.}
      
      \item{"partial" Draw the partial dependence plot for each
        predictor variable in the training data.  This is an expensive
        plot.  It might take a while to draw for large data sets or
        complex models.}

      \item{"help" show this help page in a browser}
      
    }

    If \code{y} fails to match any of the above, but it (partially) the
    name of one of the variables in the training data, then a partial
    dependence plot vs that variable is produced.
  }

  \item{burn}{
    The number of MCMC iterations to discard as burn-in.
  }

  \item{\dots}{ Additional arguments passed to the specific functions
    that do the plotting.  For residual and predicted plots that is the
    \code{\link{plot}} function.  For network structure it is
    \code{\link[igraph]{plot.igraph}}.  For partial dependence plots it
    is \code{\link{PartialDependencePlot}}.
  }

}

\details{ Residual and predicted plots should be self explanatory.  The
  network structure plot is fairly standard for neural network models.
  The width of a line linking two nodes is determined by the absolute
  value of the corresponding coefficient.

}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{BayesNnet}}
  \code{\link{PartialDependencePlot}}
}

\examples{
  ## See the examples in ?BayesNnet
}

\keyword{models}
\keyword{regression}
