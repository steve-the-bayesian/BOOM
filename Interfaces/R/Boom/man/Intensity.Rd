\name{Intensity}
\Rdversion{1.1}
\alias{Intensity}
\alias{plot.Intensity}
\title{
  Intensity function for a vector of event times
}

\description{
  Estimate and plot the intensity function describing the rate at which a
  sequence of events occurs over time.
}

\usage{
Intensity(x, ...)

\method{plot}{Intensity}(x, y = NULL, xlab = NULL, ylab = NULL,
     type = "l", zero.line = TRUE, ...)
}

\arguments{

  \item{x}{
    For \code{Intensity}: a vector of time points, given as an object of
    class \code{\link{POSIXt}} or \code{\link{Date}}.

    For \code{plot.Intensity}: an object of class \code{Intensity}, as
    produced by \code{Intensity}.
  }

  \item{y}{
    Unused.  Present to match the signature of the generic \code{\link{plot}}.
  }

  \item{xlab}{
    Character label for the horizontal axis.  If \code{NULL} then the name of
    the variable passed to \code{Intensity} is used.
  }

  \item{ylab}{
    Character label for the vertical axis.  If \code{NULL} then
    \code{"Intensity"} is used.
  }

  \item{type}{
    The type of plot to draw.  Defaults to \code{"l"} (a line plot).  See
    \code{\link{plot.default}} for other options.
  }

  \item{zero.line}{
    Logical.  If \code{TRUE} a dotted horizontal line is drawn at zero.
  }

  \item{\dots}{
    For \code{Intensity}: extra arguments passed to \code{\link{density}}.

    For \code{plot.Intensity}: extra arguments passed to \code{\link{plot}}.
  }

}

\details{
  \code{Intensity} estimates the intensity function using a kernel density
  estimate (via \code{\link{density}}) of the numeric representation of the
  event times.  Because \code{density} integrates to one, the estimate is
  rescaled by the number of events so that the area under the curve equals the
  event count.  The resulting height therefore has units of events per unit
  time.
}

\value{
  \code{Intensity} returns an object of class \code{Intensity}.  This is a
  \code{\link{density}} object whose \code{y} values have been rescaled to
  integrate to the number of events, and whose \code{x} values have been
  restored to the class (\code{POSIXt} or \code{Date}) of the input.

  \code{plot.Intensity} is called for its side effect, which is to produce a
  plot on the current graphics device.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{density}}
}

\examples{
## Simulate 500 event times clustered near the middle of a one-year window.
origin <- as.Date("2020-01-01")
event.days <- rnorm(500, mean = 180, sd = 40)
event.times <- origin + event.days

intensity <- Intensity(event.times)
plot(intensity)
}

\keyword{ hplot }
