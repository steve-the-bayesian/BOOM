\name{histabunch}
\Rdversion{1.2}
\alias{histabunch}
\title{
  A Bunch of Histograms
}
\description{

  Plot a bunch of histograms describing the marginal distributions the
  columns in a data frame.

}

\usage{
histabunch(x, gap = 1, same.scale = FALSE, boxes = FALSE,
           min.continuous = 12, max.factor = 40,
           vertical.axes = FALSE, ...)
}

\arguments{

\item{x}{A matrix or data frame containing the variables to be plotted.}
\item{gap}{The gap between the plots, measured in lines of text.}
\item{same.scale}{Logical. Indicates whether the histograms
       should all be plotted on the same scale.}
\item{boxes}{Logical. Indicates whether boxes should be drawn
  around the histograms.}

\item{min.continuous}{Numeric variables with more than \code{min.continuous}
  unique values will be plotted as continuous.  Otherwise they will
  be treated as factors.}

\item{max.factor}{Factors with more than \code{max.factor} levels will
  be beautified (ha!) by combining their remaining levels into an
  "other" category.}

\item{vertical.axes}{Logical value indicating whether the histograms
  should be given vertical "Y" axes.}

\item{\dots}{Extra arguments passed to hist (for numeric variables) or
       barplot (for factors).}
}

\value{
  Called for its side effect, which is to produce multiple histograms
    on the current graphics device.
  }
\author{
  Steven L. Scott
}
\seealso{
  \code{\link{hist}} \code{\link{barplot}}
}
\examples{
  data(airquality)
  histabunch(airquality)
}
\keyword{ hplot }
