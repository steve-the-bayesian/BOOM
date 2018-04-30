


\alias{plot.PointProcess}
\name{plot.point.process}
\title{plotting point process data}
\Rdversion{1.0}

\description{
  plotting a PointProcess
}

\usage{
  \method{plot}{PointProcess}(x,
     y = c("density", "counting", "jitter"),
     style = y,
     start = x$start,
     end = x$end,
     kernel.scale = .1,
     ...)
}

\arguments{

  \item{x}{An object of class \code{\link{PointProcess}}.}

  \item{y}{A character string indicating the style of the plot.}

  \item{style}{A synonym for \code{y}.  If both are given, \code{y} is
    ignored.}
  \item{start}{Start time for the plot.}

  \item{end}{End time for the plot.}

  \item{kernel.scale}{ A factor by which to scale the default bandwidth
         used with \code{\link{density}} when style == "density".
         Smaller values mean greater resolution.  Larger values mean
         less smoothing.}

  \item{\dots}{Extra arguments passed to \code{\link{plot}}.}

}

\value{
  Produces a plot on the current graphics device.
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
plot(process)
}

\keyword{hplot}
