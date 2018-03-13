\name{external.legend}
\Rdversion{1.0}
\alias{AddExternalLegend}
\alias{ExternalLegendLayout}

\title{
  Add an external legend to an array of plots.
}

\description{

  \code{ExternalLegendLayout} sets up a plotting region to plot a
  regular grid of points, with an optional legend on the top or right of
  the grid.

  \code{AddExternalLegend} adds a legend to a grid of plots that was set
  up using \code{ExternalLegendLayout}.

}

\usage{
ExternalLegendLayout(nrow,
                     ncol,
                     legend.labels,
                     legend.location = c("top", "right"),
                     outer.margin.lines = rep(4, 4),
                     gap.between.plots = rep(0, 4),
                     legend.cex = 1,
                     x.axis = TRUE,
                     y.axis = TRUE)

AddExternalLegend(legend.labels,
                  legend.location = c("top", "right"),
                  legend.cex =1,
                  bty = "n",
                  ...)
}

\arguments{
  \item{nrow}{The number of rows in the array of plots.}

  \item{ncol}{The number of columns in the array of plots.}

  \item{legend.labels}{The labels to be used in the legend.}

  \item{legend.location}{Specifies whether the legend should appear on
    the top or the right hand side.  It can also be \code{NULL},
    indicating that no legend is desired.}

  \item{outer.margin.lines}{A vector of length four giving the number of
    lines of text desired for the outer margins of the plot.  See the
    \code{oma} argument of \code{\link{par}}.  This can also be
    specified as a single number, to be repeated 4 times.}

  \item{gap.between.plots}{A vector of length 4 giving the number of
    lines of text to leave between grid panels.  See the \code{mar}
    argument of \code{\link{par}}.  This can also be specified as a
    single number, to be repeated 4 times.}

  \item{legend.cex}{The scale factor that will be used for legend text.
    This must match the scale factor used in add.external.legend.}

  \item{x.axis}{Will any plots have a horizontal axis?}

  \item{y.axis}{Will any plots have a vertical axis?}

  \item{bty}{Type of box to draw around the legend.  Can be "n" (for no
    box) or "o" for a box.  See \code{\link{legend}}.  }

  \item{...}{Extra arguments passed to \code{\link{legend}}.}
}

\value{\code{ExternalLegendLayout} returns the original graphical
  parameters, intended for use with \code{\link{on.exit}}.

  \code{AddExternalLegend} returns invisible \code{NULL}.
}


\author{
  Steven L. Scott
}
\seealso{
  \code{\link{legend}} \code{\link{layout}}
}

\examples{

example.plot <- function() {
  x <- rnorm(100)
  y <- rnorm(100)
  scale <- range(x, y)
  opar <- ExternalLegendLayout(nrow = 2,
                               ncol = 2,
                               legend.labels = c("foo", "bar"))
  on.exit({par(opar); layout(1)})
  hist(x, xlim = scale, axes = FALSE, main = "")
  mtext("X", side = 3, line = 1)
  box()
  plot(x, y, xlim = scale, ylim = scale, axes = FALSE)
  box()
  axis(3)
  axis(4)
  plot(y, x, xlim = scale, ylim = scale, axes = FALSE, pch = 2, col = 2)
  box()
  axis(1)
  axis(2)
  hist(y, xlim = scale, axes = FALSE, main = "")
  mtext("Y", side = 1, line = 1)
  box()
  AddExternalLegend(legend.labels = c("foo", "bar"),
                    pch = 1:2,
                    col = 1:2,
                    legend.cex = 1.5)
}

## Now call example.plot().
example.plot()

## Because of the call to on.exit(), in example.plot,
## the original plot layout is restored.
hist(1:10)
}

\keyword{ hplot }
