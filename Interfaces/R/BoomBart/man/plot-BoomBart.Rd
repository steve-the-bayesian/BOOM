\name{plot-BoomBart}
\alias{plot.BoomBart}
\title{plotting Bart models}
\description{plot method for Bart models}

\usage{
\method{plot}{BoomBart}(x, y, ...)
}

\arguments{
  \item{x}{A \code{\link{BoomBart}} model object}

  \item{y}{If \code{y} is present then the call is dispatched to
    \code{\link{BartPartialDependencePlot}}.  Otherwise it is dispatched
    to \code{\link{PlotTreeSizeDistribution}}.  If present, \code{y}
    should either unambiguously expand to a name in
    \code{colnames(x$design.matrix)}, or else it can be the position in
    the design matrix corresponding to a variable to plot.  }

  \item{...}{Extra arguments passed along to function handlers.}
}

\value{
  The return value is the return value of the function handling the call.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
