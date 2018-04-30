


\alias{plot.PointProcessDensity}
\name{plot.density.point.process}
\title{Plot the estimated intensity of a point process}
\Rdversion{1.0}

\description{
  Plot the estimated intensity of a point process
}

\usage{
  \method{plot}{PointProcessDensity}(x, main = NULL, xlab = NULL, ylab
  = "Density", type = "l", zero.line = TRUE, ...)
}

\arguments{
  \item{x}{An object of class \code{\link{PointProcessDensity}}}.
  \item{main}{Main figure title.}
  \item{xlab}{Label for horiontal axis.}
  \item{ylab}{Label for vertical axis.}
  \item{type}{Type of figure to draw.}
  \item{zero.line}{Logical.  Should a horizontal line be drawn at zero?}
  \item{\dots}{Extra arguments passed to \code{\link{plot}}.}
}

\value{
  Draws a plot on the current graphics device.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomEvents-package}}.
}

\examples{
start <- as.POSIXct("2012-04-12")
dates <- start + sort(runif(100)) * 3600 * 24 * 7
## 100 time points uniformly distributed between start and one week
## later.
process <- PointProcess(dates, start)
plot(density(process))
}
