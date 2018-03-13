\name{add.segments}
\Rdversion{1.1}
\alias{add.segments}
\alias{AddSegments}
\title{
  Function to add horizontal line segments to an existing plot
}
\description{
   Adds horizontal line segments to an existing plot.  The segments are
   centered at \code{x} with height \code{y}.  The \code{x} values are
   assumed to be equally spaced, so that \code{diff(x)} is a constant
   '\code{dx}'.  The line segments go from \code{x +/- half.width.factor
   *dx}, so if \code{half.width.factor=.5} there will be no gaps between
   segments.  The default is to leave a small gap.

   This function was originally used to add reference lines to
   side-by-side boxplots.
}
\usage{
AddSegments(x, y, half.width.factor = 0.45, ...)
}
\arguments{
  \item{x}{
    A numeric vector giving the midpoints of the line segments.
}
  \item{y}{
    A numeric vector of the same length as \code{x} giving the vertical
   position of the line segments
}
  \item{half.width.factor}{
    See 'description' above.
}
  \item{\dots}{
    graphical parameters controlling the type of lines used in the line segments
}
}
\value{
  Called for its side effect.
}
\author{
  Steven L. Scott
}

\seealso{
  \code{\link{boxplot.true}}
}
\examples{

x <- rnorm(100)
y <- rnorm(100, 1)
boxplot(list(x=x,y=y))
AddSegments(1:2, c(0, 1))  ## add segments to the boxplot

}
\keyword{ aplot }
