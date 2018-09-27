\name{circles}
\title{Draw Circles}
\alias{circles}

\description{Draw circles on the current graphics device.}

\usage{
circles(center, radius, ...) 
}

\arguments{
  \item{center}{A two-column matrix giving the coordinates of the circle center.
   If a single circle is to be drawn then a 2-element vector can be passed
   instead.}

  \item{radius}{The radii of the circles.  A scalar value will be
  repeated if \code{center} is a matrix with more than one row.  }

  \item{\dots}{Extra arguments passed to 'segments'.  See
    \code{\link{par}} for options controlling line type, line width,
    color, etc.}
}

\value{
  Returns invisible \code{NULL}.
}

\details{ Draws circles on the current graphics device.  This is a
  low-level plotting function similar to \code{\link{points}},
  \code{\link{lines}}, \code{\link{segments}}, etc.  }

\examples{
  plot(1:10, type = "n")
  circles(cbind(c(2, 3, 4), c(4, 5,6 )), radius = c(.3, .4, .5))
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
