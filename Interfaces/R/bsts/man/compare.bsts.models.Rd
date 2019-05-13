% Copyright 2018 Google Inc. All Rights Reserved.
%
% This library is free software; you can redistribute it and/or
% modify it under the terms of the GNU Lesser General Public
% License as published by the Free Software Foundation; either
% version 2.1 of the License, or (at your option) any later version.
%
% This library is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% Lesser General Public License for more details.
%
% You should have received a copy of the GNU Lesser General Public
% License along with this library; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA

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
                  grid = TRUE,
                  cutpoint = NULL)
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

  \item{cutpoint}{Either \code{NULL}, or an integer giving the
    observation number used to define a holdout sample.  Prediction
    errors occurring after the cutpoint will be true out of sample
    errors.  If \code{NULL} then all prediction errors are "in sample".
    See the discussion in \code{\link{bsts.prediction.errors}}.}
}

\value{
  Invisibly returns the matrix of cumulative one-step ahead prediction
  errors (the lines in the top panel of the plot).  Each row in the
  matrix corresponds to a model in model.list.
}

\author{
  Steven L. Scott  \email{steve.the.bayesian@gmail.com}
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  trend.only <- bsts(y, ss, niter = 250)

  ss <- AddSeasonal(ss, y, nseasons = 12)
  trend.and.seasonal <- bsts(y, ss, niter = 250)

  CompareBstsModels(list(trend = trend.only,
                         "trend and seasonal" = trend.and.seasonal))

  CompareBstsModels(list(trend = trend.only,
                         "trend and seasonal" = trend.and.seasonal),
                          cutpoint = 100)

}

\keyword{chron}
