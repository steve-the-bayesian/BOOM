\name{compare.many.densities}
\Rdversion{1.0}
\alias{compare.many.densities}
\alias{CompareManyDensities}

\title{
  Compare several density estimates.
}
\description{
  Produce a plot that compares the kernel density estimates for
  each element in a series of Monte Carlo draws of a vector or
  matrix.
}
\usage{
CompareManyDensities(list.of.arrays,
                     style = c("density", "box"),
                     main = "",
                     color = NULL,
                     gap = 0,
                     burn = 0,
                     suppress.labels = FALSE,
                     x.same.scale = TRUE,
                     y.same.scale = FALSE,
                     xlim = NULL,
                     ylim = NULL,
                     legend.location = c("top", "right"),
                     legend.cex = 1,
                     reflines = NULL,
                     ...)
}

\arguments{

  \item{list.of.arrays}{A list of arrays representing the MCMC draws of
    the vector or matrix in question.  Each list element represents a
    different group.  The first index in each list list element
    represents the Monte Carlo draw number (or iteration).  The
    remaining indices represent the variables to be plotted.  If the
    first list element has variable names assigned to its indices,
    these will be used to label the plots.  }

  \item{style}{The style of plot to use for comparing distributions.}

  \item{main}{The main title of the plot.}

  \item{color}{A vector of colors to be used for representing the groups.}
  \item{gap}{The gap (in lines) between plots.}
  \item{burn}{The number of MCMC iterations to be discarded as burn-in.}

  \item{suppress.labels}{Logical.  If \code{FALSE} then the dimnames (if
    any) of the first element in \code{list.of.arrays} will be used
    to annotate the plot.  If \code{TRUE} then no labels will be
    used. }

  \item{x.same.scale}{Logical indicating whether the same horizontal scale
    should be used for all the plots.  }

  \item{y.same.scale}{Logical indicating whether the same vertical scale
    should be used for all the plots.  This argument is ignored if
    \code{style == "box"}.}

  \item{xlim}{Either \code{NULL}, or a pair of numbers giving limits for
    the horizontal axis.  If \code{xlim} is set then the same xlim
    values will be used for all plots and the \code{x.same.scale}
    argument will be ignored.  }

  \item{ylim}{Either \code{NULL}, or a pair of numbers giving limits for
    the vertical axis.  If \code{ylim} is set then the same ylim
    values will be used for all plots and the \code{y.same.scale}
    argument will be ignored.  This argument is ignored if
    \code{style == "box".}}

  \item{legend.location}{The location of the legend, either on top or
    at the right.  It can also be \code{NULL} in which case no legend
    will appear.  The legend names will be taken from
    names(list.of.arrays).  If it does not have names, then no
    legend will be produced.}

  \item{legend.cex}{The relative scale factor to use for the legend
    text.}

  \item{reflines}{This can be \code{NULL}, in which case no reference
    lines are drawn, it can be a single real number in which case a
    reference line will be drawn at that value in each panel, or it can
    be a vector with length equal to the number of panels, in which case
    a reference line will be drawn at each panel-specific value.}

  \item{\dots}{Extra arguments passed to \code{\link{CompareDen}}.}
}

\author{
  Steven L. Scott
}
\seealso{
  \code{\link{density}}, \code{\link{CompareManyTs}}
}

\examples{
x <- array(rnorm(9000), dim = c(1000, 3, 3))
dimnames(x) <- list(NULL, c("Larry", "Moe", "Curly"), c("Larry", "Eric", "Sergey"))
y <- array(rnorm(9000), dim = c(1000, 3, 3))
z <- array(rnorm(9000), dim = c(1000, 3, 3))
data <- list(x = x, y = y, z = z)
CompareManyDensities(data, color = c("red", "blue", "green"))
CompareManyDensities(data, style = "box")

x <- matrix(rnorm(5000), nrow = 1000)
colnames(x) <- c("Larry", "Moe", "Curly", "Shemp", "???")
y <- matrix(rnorm(5000), nrow = 1000)
z <- matrix(rnorm(5000), nrow = 1000)
data <- list(x = x, y = y, z = z)
CompareManyDensities(data, color = c("red", "blue", "green"))
CompareManyDensities(data, style = "box")
}

\keyword{ hplot }
