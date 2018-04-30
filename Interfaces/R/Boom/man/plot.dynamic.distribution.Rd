\name{plot.dynamic.distribution}
\Rdversion{1.0}
\alias{PlotDynamicDistribution}
\title{
  Plots the pointwise evolution of a distribution over an index set.
}

\description{ Produces an dynamic distribution plot where gray scale
  shading is used to show the evolution of a distribution over an index
  set.  This function is particularly useful when the index set is too
  large to do side-by-side boxplots.  }

\usage{
PlotDynamicDistribution(curves,
                        timestamps = NULL,
                        quantile.step=.01,
                        xlim = NULL,
                        xlab = "Time",
                        ylim = range(curves, na.rm = TRUE),
                        ylab = "distribution",
                        add = FALSE,
                        axes = TRUE,
                        ...)
}
\arguments{

  \item{curves}{ A matrix where each row represents a curve (e.g. a
    simulation of a time series from a posterior distribution) and
    columns represent different points in the index set.  For example, a
    long time series would be a wide matrix.  }

  \item{timestamps}{ An optional vector of "time stamps" that
    \code{curves} will be plotted against.  The length of
    \code{timestamps} must match the number of columns in \code{curves}.
    If \code{timestamps} is \code{NULL} then the function attempts to
    extract time stamps from the \code{colnames(curves)}.  If no
    appropriate time stamps can be found then the positive integers will
    be used as time stamps.}

  \item{quantile.step}{Each color step in the plot corresponds to this
    difference in quantiles.  Smaller values make prettier plots, but
    the plots take longer to produce.}

  \item{xlim}{The x limits (x1, x2) of the plot.  Note that \code{ x1 >
    x2 } is allowed and leads to a "reversed axis".}

  \item{xlab}{Label for the horzontal axis. }

  \item{ylim}{The y limits (y1, y2) of the plot.  Note that \code{ y1 >
    y2 } is allowed and leads to a "reversed axis".}

  \item{ylab}{Label for the vertical axis. }

  \item{add}{Logical.  If true then add the plot to the current plot.
    Otherwise a fresh plot will be created.}

  \item{axes}{Logical.  Should axes be added to the plot?}

  \item{\dots}{Extra arguments to pass on to \code{\link{plot}} }

}

\details{ The function works by passing many calls to
  \code{\link{polygon}}.  Each polygon is associated with a quantile
  level, with darker shading near the median. }

\value{ This function is called for its side effect, which is to produce
  a plot on the current graphics device.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
\examples{
  x <- t(matrix(rnorm(1000 * 100, 1:100, 1:100), nrow=100))
  ## x has 1000 rows, and 100 columns.  Column i is N(i, i^2) noise.

  PlotDynamicDistribution(x)
  time <- as.Date("2010-01-01", format = "\%Y-\%m-\%d") + (0:99 - 50)*7
  PlotDynamicDistribution(x, time)
}

\keyword{ hplot }
