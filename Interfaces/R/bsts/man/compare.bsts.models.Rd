\name{compare.bsts.models}

\alias{CompareBstsModels}

\Rdversion{1.0}

\title{
  Compare bsts models
}

\description{ Produce a set of line plots showing the cumulative
   absolute one step ahead prediction errors for different models.  This
   plot not only shows which model is doing the best job predicting the
   data, it highlights regions of the data where the predictions are
   particularly good or bad.
 }

\usage{
CompareBstsModels(model.list,
                  burn = SuggestBurn(.1, model.list[[1]]),
                  filename = "",
                  colors = NULL,
                  lwd = 2,
                  xlab = "Time",
                  main = "",
                  grid = TRUE)
}

\arguments{
  \item{model.list}{A list of \code{\link{bsts}} models.}

  \item{burn}{The number of initial MCMC iterations to remove from each
    model as burn-in.}

  \item{filename}{ A string.  If non-empty string then a pdf of the plot
    will be saved in the specified file.}

  \item{colors}{A vector of colors to use for the different lines in the
    plot.  If \code{NULL} then the \code{\link{rainbow}} pallette will
    be used.  }

  \item{lwd}{The width of the lines to be drawn.}

  \item{xlab}{Label for the horizontal axis.}

  \item{main}{Main title for the plot.}

  \item{grid}{Logical.  Should gridlines be drawn in the background?}
}

\value{
  Invisibly returns the matrix of cumulative one-step ahead prediction
  errors (the lines in the top panel of the plot).  Each row in the
  matrix corresponds to a model in model.list.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  trend.only <- bsts(y, ss, niter = 500)

  ss <- AddSeasonal(ss, y, nseasons = 12)
  trend.and.seasonal <- bsts(y, ss, niter = 500)

  CompareBstsModels(list(trend = trend.only,
                         "trend and seasonal" = trend.and.seasonal))
}

\keyword{chron}
