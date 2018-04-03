\name{plot.many.ts}
\Rdversion{1.1}
\alias{plot.many.ts}
\alias{PlotManyTs}
\title{
  Multiple time series plots
}
\description{
  Plots many time series plots on the same graphical device.  Each plot
  gets its own frame.  Scales can be adjusted to see variation in each
  plot (each plot gets its own scale), or variation between plots
  (common scale).
}
\usage{
PlotManyTs(x, type = "l", gap = 0, boxes = TRUE, truth = NULL,
           thin = 1, labs, same.scale = TRUE, ylim = NULL,
           refline = NULL, color = NULL, ...)
}

\arguments{

  \item{x}{Matrix, data frame, list, or 3-dimensional array to be plotted. }

  \item{type}{ type of line plots to produce.  See
    \code{\link{plot.default}} for other options.  }

  \item{gap}{Number of lines of space to put between plots.  }

  \item{boxes}{Logical indicating whether boxes should be drawn around
    each plot.  }

  \item{truth}{A vector or matrix of reference values to be added to
    each plot as a horizontal line.  The dimension should match
    \code{dim(x)[-1]} }

  \item{thin}{Frequency of observations to plot.  E.g. \code{thin=10}
    means plot every 10'th observation.  Thinning can speed things up
    when plotting large amounts MCMC output.  }

  \item{labs}{Optional character vector giving the title (e.g. variable
    name) for each plot.  If \code{labs} is missing then column names or
    dimnames will be used to label the plots.  If \code{labs} is
    \code{NULL} then no labels will be used. }

  \item{same.scale}{Logical indicating whether plots should be drawn
    with a common vertical axis, which is displayed on alternating rows
    of the plot.  If \code{FALSE} then the vertical axis of each plot is
    scaled to the range of data in that plot, but no tick marks are
    displayed. }

  \item{ylim}{Scale of the vertical axis.  If non-NULL then same.scale
    is set to \code{TRUE} and the supplied scale is used for all plots.
    }

  \item{refline}{ a vector or scalar value to use as a reference line.
    This is a supplement to the \code{truth} argument.  It can be useful
    when comparing true values (used in a simulation), estimated values
    (e.g. point estimates of parameters) and MCMC output.  }

  \item{color}{ Vector of colors to use in the plots.  }

  \item{\dots}{ Extra arguments passed to \code{\link{plot}} and
    \code{\link{axis}}.  }

}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{plot.ts}} (for plotting a small number of time series)
  \code{\link{plot.macf}}
}

\examples{

x <- matrix(rnorm(1000), ncol = 10)
PlotManyTs(x)
PlotManyTs(x, same = FALSE)

}

\keyword{ hplot }
