\name{compare.many.ts}
\Rdversion{1.0}
\alias{compare.many.ts}
\alias{CompareManyTs}

\title{
  Compares several density estimates.
}
\description{
  Produce a plot that compares the kernel density estimates for
  each element in a series of Monte Carlo draws of a vector or
  matrix.
}
\usage{
CompareManyTs(list.of.ts, burn = 0, type = "l", gap = 0,
              boxes = TRUE, thin = 1, labels = NULL,
              same.scale = TRUE, ylim = NULL, refline = NULL,
              color = NULL, ...)
}

\arguments{

\item{list.of.ts}{A list of time series matrices, data.frames or
  3-dimensional arrays, all of the same size.  The list elements
  correspond to groups.  The first index of the array in each list
  element corresponds to time.  The subsequent indices correspond to
  variables to be plotted.  }

\item{burn}{The number of initial observations to be discarded as
       burn-in (when plotting MCMC output). }

\item{type}{The plotting type to use when plotting the time series.  See
  \code{\link{plot}}.}

\item{gap}{The amount of space to put between plots.}

\item{boxes}{Logical.  Should boxes be drawn around the plots?}

\item{thin}{Plot every thin'th observation.  This can reduce the
       amount of time it takes to make the plot if there are many long
       time series.
}

\item{labels}{A character vector to use as labels for individual plots.}

\item{same.scale}{Logical.  If TRUE then all plots are shown on the
       same verical scale, and vertical axes are drawn.  If FALSE
       then each plot gets its own scale.
}

\item{ylim}{The scale of the vertical axis.  If non-NULL then
       same.scale will be set to TRUE.
}

\item{refline}{The scalar value at which a thin dotted horizontal
       line should be plotted in each panel.  This is useful for
       highlighting zero, for example.
}

\item{color}{A vector of colors to use for the plots.}

\item{\dots}{Extra arguments passed to 'plot' and 'axis'.}

}

\author{
  Steven L. Scott
}
\seealso{
  \code{\link{PlotManyTs}}, \code{\link{CompareManyDensities}}
}

\examples{
x <- array(rnorm(9000), dim = c(1000, 3, 3))
dimnames(x) <- list(NULL, c("Larry", "Moe", "Curly"), c("Larry", "Eric", "Sergey"))
y <- array(rnorm(9000), dim = c(1000, 3, 3))
z <- array(rnorm(9000), dim = c(1000, 3, 3))
data <- list(x = x, y = y, z = z)
CompareManyTs(data, color = c("red", "blue", "green"))

x <- matrix(rnorm(5000), nrow = 1000)
colnames(x) <- c("Larry", "Moe", "Curly", "Shemp", "???")
y <- matrix(rnorm(5000), nrow = 1000)
z <- matrix(rnorm(5000), nrow = 1000)
data <- list(x = x, y = y, z = z)
CompareManyTs(data, color = c("red", "blue", "green"))
}

\keyword{ hplot }
