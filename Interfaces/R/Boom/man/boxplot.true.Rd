\name{boxplot.true}
\Rdversion{1.1}
\alias{boxplot.true}
\alias{BoxplotTrue}
\title{
  side-by-side boxplots from a matrix, with optional reference values
}
\description{
  Plots side-by-side boxplots of the columns of the matrix \code{x}.
  Each boxplot can have its own reference line (\code{truth}) and
  standard error lines \code{se.truth}, if desired.  This function was
  originally written to display MCMC output, where the reference lines
  were true values used to test an MCMC simulation.
}
\usage{
BoxplotTrue(x, truth = NULL, vnames = NULL, center = FALSE,
            se.truth = NULL, color = "white", ...)
}

\arguments{
  \item{x}{
    The matrix whose columns are to be plotted.
}
  \item{truth}{
    (optional) A vector of reference values with length equal to \code{ncol(x)}.
}
  \item{vnames}{
    (optional) character vector giving the column names of \code{x}.
}
  \item{center}{
    (optional) logical.  If \code{truth} is supplied then
  \code{center=TRUE} will center each column of \code{x} around
  \code{truth} to show the variation around the reference line.
}
  \item{se.truth}{
    (optional) numeric vector of length \code{ncol(x)}.  If \code{truth}
  is supplied then additional reference lines will be drawn at
  \code{truth +/- 2*se.truth}.
}
  \item{color}{
    (optional) vector of colors for each boxplot.
}
  \item{\dots}{
    additional arguments to \code{\link{boxplot}}.
}
}
\value{
  called for its side effect
}
\author{
  Steven L. Scott
}
\seealso{
  \code{\link{boxplot.matrix}},
  \code{\link{boxplot}},
}
\examples{
x <- t(matrix(rnorm(5000, 1:5, 1:5), nrow=5))
BoxplotTrue(x, truth=1:5, se.truth=1:5, col=rainbow(5), vnames =
  c("EJ", "TK", "JT", "OtherEJ", "TJ") )
}
\keyword{ hplot }
