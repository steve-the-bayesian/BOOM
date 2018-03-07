\name{boxplot.mcmc.matrix}
\Rdversion{1.1}
\alias{boxplot.mcmc.matrix}
\alias{BoxplotMcmcMatrix}
\title{
  Plot the distribution of a matrix
}
\description{
  Plot the marginal distribution of each element in the Monte Carlo
  distribution of a matrix (e.g. a variance matrix or
  transition probability matrix).  Rows and columns in the boxplots
  correspond to rows and columns in the matrix being plotted.
}
\usage{
  BoxplotMcmcMatrix(X, ylim = range(X), col.names,
                    row.names, truth, colors = NULL,
                    las = 0, ...)
}

\arguments{
  \item{X}{
    3 dimensional array.  The first dimension is the Monte Carlo index
  (e.g. MCMC iteration).  The second and third dimensions are the row
  and column of the matrix being plotted.  E.g. \code{X[i,j,k]} is Monte
  Carlo draw \code{i} of matrix element \code{j,k}.
}
  \item{ylim}{
    2-vector giving the lower and upper limits of the vertical axis.
}
  \item{col.names}{
    (optional) character vector giving the names of matrix columns
  (third dimension of \code{X}).
}
  \item{row.names}{
    (optional) character vector giving the names of matrix rows
    (second dimension of \code{X}).
}
\item{truth}{
  (optional) scalar or matrix giving the values of reference lines to
  be plotted on each boxplot.  If a scalar then the same value will be
  used for each boxplot.  If a matrix then the rows and columns of the
  matrix correspond to the second and third dimension of \code{X}.
}
\item{colors}{
  A vector of colors to use for the boxplots.  Each row uses the same
  color scheme.
}

\item{las}{Controls the orientation of axis labels.  See the \code{las}
  section in the help page for \code{\link{par}}.
}

\item{\dots}{Extra arguments passed to \code{\link{boxplot}}}

}
\value{
  Called for its side effect, which is to draw a set of side-by-side
  boxplots on the current graphics device.
}
\author{
  Steven L. Scott
}

\seealso{
  \code{\link{boxplot.true}},
  \code{\link{boxplot}}
}

\examples{
  X <- array(rnorm(1000 * 3 * 4), dim=c(1000, 3, 4))
  dimnames(X)[[2]] <- paste("row", 1:3)
  dimnames(X)[[3]] <- paste("col", 1:4)
  BoxplotMcmcMatrix(X)

  truth <- 0
  BoxplotMcmcMatrix(X, truth=truth)

  truth <- matrix(rnorm(12), ncol=4)
  BoxplotMcmcMatrix(X, truth=truth)
}

\keyword{ hplot }
