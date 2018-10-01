% Copyright 2018 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{partial.dependence.plot}
\Rdversion{1.1}
\alias{PartialDependencePlot}

\title{
  Plot a Bayesian Neural Network
}

\description{
  Plot the relationship between Y and a single X variable, averaging
  over the values of the other X's.
}

\usage{
  PartialDependencePlot(model,
                        which.variable,
                        burn = SuggestBurn(model),
                        data.fraction = .2,
                        gridsize = 50,
                        mean.only = FALSE,
                        show.points = TRUE,
                        xlab = NULL,
                        ylab = NULL,
                        ylim = NULL,
                        report.time = FALSE,
                        ...) 
}

\arguments{
  \item{model}{ An object of class \code{BayesNnet}. }

  \item{which.variable}{ Either an integer denoting the position of the
    X variable in the data frame used to fit the model, or a character
    string naming that variable.  }
  
  \item{burn}{
    The number of MCMC iterations to discard as burn-in.
  }

  \item{data.fraction}{The fraction of observations in the predictor
    matrix to use when constructing the partial dependence plot.  A
    random sub-sample of this fraction will be taken (without
    replacement) for the purposes of marginalizing over the remaining
    predictors.  }

  \item{gridsize}{The number of grid points to use on the X axis.}
  
  \item{mean.only}{ Logical.  If \code{TRUE} then only the mean is
    plotted at each point.  If \code{FALSE} then the posterior of the
    function value is plotted.
  }
     
  \item{show.points}{If \code{TRUE} then the scatterplot of x vs y is
    added to the graph.  Otherwise the points are left off.  Note that
    the estimated function might not match the pattern in the
    scatterplot, because the points in the scatterplot are not adjusted
    for the values of the other X variables.  }
  
  \item{xlab}{Label for the X axis.  NULL produces a default label.
    Use "" for no label.}
  
  \item{ylab}{ Label for the Y axis.  NULL produces a default label.
    Use "" for no label.  }

  \item{ylim}{Limits on the vertical axis.  If NULL then the plot will
    default to its natural vertical limits.}

  \item{report.time}{Print the time required to produce the plot.}
  
  \item{\dots}{Extra arguments are passed either to 'plot' (if mean.only
    is \code{TRUE})' or 'PlotDynamicDistribution' (otherwise).}
}

\details{
  A partial dependence plot shows the relationship between Y and a
  single X variable, averaging over the values of the other X's in a
  possibly nonlinear regression model.  Partial dependence plots are a
  generalization of the "added variable plot" idea from linear
  regression models.

  A partial dependence plot is more expensive to produce than most other
  plots, because a set of predictions must be generated at each point on
  the X axis.  This is done by taking a random subset of the training
  data, and evaluating the posterior predictive distribution with each
  observation's target X value set to each value of X on the grid.

}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{plot.BayesNnet}}
}

\examples{
  # Please see the code in ?BayesNnet
}

