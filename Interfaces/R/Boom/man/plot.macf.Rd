\name{plot.macf}
\Rdversion{1.1}
\alias{plot.macf}
\alias{PlotMacf}
\title{
  Plots individual autocorrelation functions for many-valued time series
}
\description{
  Produces individual autocorrelation functions for many-valued time
  series such as those produced by highly multivariate MCMC output.
  Cross-correlations such as those produced by \code{\link{acf}} are not shown.
}
\usage{
PlotMacf(x, lag.max = 40, gap = 0.5, main = NULL, boxes = TRUE,
         xlab = "lag", ylab = "ACF", type = "h")
}
\arguments{
  \item{x}{
    matrix or 3-way array of MCMC output (or other time series).  The
    first dimension represents discrete time.
}
  \item{lag.max}{
    maximum lag to use when computing ACF's.
}
  \item{gap}{
    non-negative scalar. gap between plots
}
  \item{main}{
    character.  main title for the plot
}
  \item{boxes}{
    logical.  Should boxes be drawn around the plots
}
  \item{xlab}{
    character label for horizontal axis.
}
  \item{ylab}{
    character label for vertical axis.
}
  \item{type}{
    type of line plot to show.  Defaults to "h".  See
    \code{\link{plot.default}} for other options.
}

}
\value{
  Called for its side effect
}
\author{
  Steven L. Scott
}

\seealso{
  \code{\link{acf}},
  \code{\link{plot.many.ts}}.
}
\examples{
x <- matrix(rnorm(1000), ncol=10)
PlotMacf(x)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{ hplot }
