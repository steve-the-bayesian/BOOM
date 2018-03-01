\name{plot.density.contours}
\Rdversion{1.0}
\alias{PlotDensityContours}

\title{
  Contour plot of a bivariate density.
}

\description{

  Contour plot of one ore more bivariate densities.  This function was
  originally created to implement PairsDensity, but might be useful on
  its own.

}
\usage{
PlotDensityContours(draws,
                    x.index = 1,
                    y.index = 2,
                    xlim = NULL,
                    ylim = NULL,
                    nlevels = 20,
                    subset = NULL,
                    color = NULL,
                    lty = NULL,
                    axes = TRUE,
                    ...)

}

\arguments{

  \item{draws}{Either a matrix or a list of matrices.  If a list is
    provided then each list element is plotted as a separate set of
    contours, and all matrices must have the same number of columns
    (though the number of rows can differ).}

  \item{x.index}{The index of the parameter to plot on the
    horizonal axis.}

  \item{y.index}{The index of the beta coefficient to plot on the
    vertical axis.}

  \item{xlim}{Limits on the horizontal axis.  If NULL then the plot is
    just wide enough to fit the contours.}

  \item{ylim}{Limits on the vertical axis.  If NULL then the plot is
    just tall enough to fit the contours.}

  \item{nlevels}{The number of contour levels to plot.}

  \item{subset}{If draws is a list, then this can be a numerical vector.
    If draws has names, then subset can be a character vector naming
    which elements to include.  If \code{NULL} then all elements of
    \code{draws} are plotted.}

  \item{color}{The color to use for different elemetns in \code{draws}.}

  \item{lty}{The line types to use for the different elements in
    \code{draws}.}

  \item{axes}{Logical. Should x and y axies be drawn?}

  \item{...}{Extra arguments passed to \code{\link{contour}}.}
}

\author{
  Steven L. Scott
}
\seealso{
  \code{\link{contour}}, \code{\link[MASS]{kde2d}}
}

\examples{
## You can see the pairs plot for a single set of draws.
y <- matrix(rnorm(5000, mean = 1:5), ncol = 5, byrow = TRUE)
PlotDensityContours(y, 3, 1)

## You can also compare two or more sets of draws.
z <- matrix(rnorm(2500, mean = 2:6), ncol = 5, byrow = TRUE)
PlotDensityContours(list("first set" = y, "second set" = z), 3, 1)
}

\keyword{ hplot }
