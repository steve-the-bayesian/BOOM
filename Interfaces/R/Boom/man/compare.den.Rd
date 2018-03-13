\name{compare.den}
\Rdversion{1.2}
\alias{compare.den}
\alias{CompareDen}
\alias{compare.densities}
\alias{CompareDensities}
\title{
  Compare several density estimates.
}
\description{
  Produces multiple density plots on a single axis, to compare the
  columns of a matrix or the elements of a list.
}
\usage{
CompareDensities(x,
                 legend.text = NULL,
                 legend.location = "topright",
                 legend.title = NULL,
                 xlim = NULL,
                 ylim = NULL,
                 xlab = "parameter",
                 ylab = "density",
                 main = "",
                 lty = NULL,
                 col = "black",
                 axes = TRUE,
                 na.rm = TRUE,
                 ...)
}
\arguments{

  \item{x}{ matrix or list of numeric vectors.  A density plot is
    produced for each column of the matrix or element of the list.  }

  \item{legend.text}{ (optional) character vector giving names of each
    density plot.  }

  \item{legend.location}{Entry that can be passed to
    \code{\link{legend}}.  }

  \item{legend.title}{ The legend title.  }

  \item{xlim}{ (optional) horizonal range of the plotting region.  If
    omitted the region will be sized to fit all the observations in
    \code{x}. }

  \item{ylim}{ (optional) vertical range of the plotting region.  If
    omitted the region will be sized to fit all empirical density plots.
    }

  \item{xlab}{ label to be placed on the horizontal axis }

  \item{ylab}{ label to be placed on the vertical axis }

  \item{main}{ main title for the plot }

  \item{lty}{The line types to use for the different densities.  See
    \code{\link{par}}.  If \code{NULL} then a different line type will
    be used for each density.}

  \item{col}{ vector of colors for the densities to be plotted.  }

  \item{axes}{Logical.  Should axes and a box be drawn around the
    figure?}

  \item{na.rm}{Logical value indicating whether \code{NA}'s should be
    removed.}

\item{\dots}{ Other graphical parameters passed to
  \code{\link{plot.density}}, and \code{\link{lines}}.  }

}

\value{
  Called for its side effect, which is to produce multiple density plots
    on the current graphics device.
  }
\author{
  Steven L. Scott
}
\seealso{
  \code{\link{density}}
}
\examples{

x <- t(matrix(rnorm(5000, 1:5, 1:5), nrow=5))
CompareDensities(x, legend.text=c("EJ", "TK", "JT", "OtherEJ", "TJ"),
                 col=rainbow(5), lwd=2)
}
\keyword{ hplot }
