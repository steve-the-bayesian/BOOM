\name{compare.dynamic.distributions}
\Rdversion{1.0}
\alias{CompareDynamicDistributions}

\title{
  Compare Dynamic Distributions
}
\description{
   Produce a plot showing several stacked dynamic distributions over the same
   horizontal axis.
}
\usage{
CompareDynamicDistributions(
    list.of.curves,
    timestamps,
    style = c("dynamic", "boxplot"),
    xlab = "Time",
    ylab = "",
    frame.labels = rep("", length(list.of.curves)),
    main = "",
    actuals = NULL,
    col.actuals = "blue",
    pch.actuals = 1,
    cex.actuals = 1,
    vertical.cuts = NULL,
    ...)
}

\arguments{

 \item{list.of.curves}{A list of matrices, all having the same number of
     columns.  Each matrix represents a distribution of curves, with
     rows corresponding to individual curves, and columns to time
     points.}

 \item{timestamps}{A vector of time stamps, with length matching the
     number of columns in each element of list.of.curves.}

 \item{style}{Should the curves be represented using a dynamic
     distribution plot, or boxplots.  Boxplots are better for small
     numbers of time points.  Dynamic distribution plots are better for
     large numbers of time points.}

 \item{xlab}{Label for the horizontal axis.}

 \item{ylab}{Label for the (outer) vertical axis.}

 \item{frame.labels}{Labels for the vertical axis of each subplot. The
     length must match the number of plot.}

 \item{main}{Main title for the plot.}

 \item{actuals}{If non-\code{NULL}, actuals should be a numeric vector
     giving the actual "true" value at each time point.}

 \item{col.actuals}{Color to use for the actuals.  See \code{\link{par}}.}

 \item{pch.actuals}{Plotting character(s) to use for the actuals.  See
   \code{\link{par}}.}

 \item{cex.actuals}{Scale factor for actuals.  See \code{\link{par}}.}

 \item{vertical.cuts}{If non-\code{NULL} then this must be a vector of
     the same type as \code{timestamps} with length matching the number
     of plots.  A vertical line will be drawn at this location for each
     plot.  Entries with the value \code{NA} signal that no vertical
     line should be drawn for that entry.}

 \item{\dots}{Extra arguments passed to
     \code{\link{PlotDynamicDistribution}} or
     \code{\link{TimeSeriesBoxplot}}.}

 }

\author{
  Steven L. Scott
}

\keyword{ hplot }
