\name{pairs.density}
\Rdversion{1.0}
\alias{PairsDensity}

\title{
  Pairs plot for posterior distributions.
}

\description{

  A pairs plot showing the posterior distribution of the given list of
  Monte Carlo draws.  Plots above the diagonal show the posterior
  distribution on a scale just wide enough to fit the plots.  The
  diagonal shows a marginal density plot, and the subdiagonal shows the
  distribution with all plots on a common scale.

}
\usage{
PairsDensity(draws,
             nlevels = 20,
             lty = NULL,
             color = NULL,
             subset = NULL,
             labels,
             legend.location = "top",
             legend.cex = 1,
             label.cex = 1,
             ...)
}

\arguments{


  \item{draws}{Either a matrix or a list of matrices.  If a list is
    provided then each list element is plotted as a separate set of
    contours, and all matrices must have the same number of columns
    (though the number of rows can differ).}

  \item{nlevels}{The number of contour levels to plot.}

  \item{lty}{The line types to use for the different elements in
    \code{draws}.}

  \item{color}{The color to use for different elements in \code{draws}.}

  \item{subset}{If draws is a list, then this can be a numerical vector.
    If draws has names, then subset can be a character vector naming
    which elements to include.  If \code{NULL} then all elements of
    \code{draws} are plotted.}

  \item{labels}{If \code{labels} is missing and the first element of
    draws has non-\code{NULL} \code{colnames} then these will be used to
    label the pairs plot.  If a character vector of length
    \code{ncol(draws[[1]])} then this character vector will be used in
    place of the colnames.  If \code{NULL} then no labels will be used.}

  \item{legend.location}{Either \code{"top"}, or \code{"right"}
    specifying the location for the legend, or \code{NULL}, indicating
    that no legend is desired.  if draws is a matrix or a singleton list
    then no legend is produced.}

  \item{legend.cex}{Scale factor to use for the legend labels.}

  \item{label.cex}{Scale factor to use for the row and column labels.}

  \item{...}{Extra arguments (graphical parameters), passed to
      \code{\link{plot}}, \code{\link{PlotDensityContours}},
      \code{\link{axis}}, and \code{\link{AddExternalLegend}}.}

}

\author{
  Steven L. Scott
}
\seealso{
  \code{\link{pairs}}, \code{\link{CompareDensities}}, \code{\link{CompareManyDensities}}
}

\examples{

## You can see the pairs plot for a single set of draws.
y <- matrix(rnorm(5000, mean = 1:5), ncol = 5, byrow = TRUE)
PairsDensity(y)

## You can also compare two or more sets of draws.
z <- matrix(rnorm(2500, mean = 2:6), ncol = 5, byrow = TRUE)
PairsDensity(list("first set" = y, "second set" = z))

}

\keyword{ hplot }
