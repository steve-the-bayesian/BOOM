% Copyright 2019 Steven L. Scott All Rights Reserved.
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

\name{plot.mbsts}
\title{Plotting Functions for Multivariate Bayesian Structural Time Series}

\alias{plot.mbsts}
\alias{PlotMbstsSeriesMeans}

\description{Functions to plot the results of a model fit using
  \code{\link{mbsts}}.}

\usage{

\method{plot}{mbsts}(x, y = c("means", "help"), ...)

PlotMbstsSeriesMeans(mbsts.object,
                     series.id = NULL,
                     same.scale = TRUE,
                     burn = SuggestBurn(.1, mbsts.object),
                     time, 
                     show.actuals = TRUE,
                     ylim = NULL,
                     gap = 0, 
                     ...)
}

\arguments{
  \item{x}{An object of class \code{\link{mbsts}}.}

  \item{y}{A character string indicating the aspect of the model that
    should be plotted.}

  \item{mbsts.object}{An object of class \code{\link{mbsts}}.}

  \item{series.id}{Indicates which series should be plotted.  An
    integer, logical, or character vector.}

  \item{same.scale}{Logical.  If \code{TRUE} then all the series or
    state components will be plotted with the same scale on the vertical
    axis.  If \code{FALSE} then each plot will get its own scale for the
    vertical axis.}

  \item{burn}{The number of MCMC iterations to discard as burn-in.}

  \item{time}{An optional vector of values to plot against.  If missing,
    the default is to diagnose the time scale of the original time
    series.}

  \item{show.actuals}{Logical.  If \code{TRUE} then actual values from
    the fitted series will be shown on the plot.}
  
  \item{ylim}{Limits for the vertical axis.  If \code{NULL} these will
    be inferred from the state components and the \code{same.scale}
    argument.  Otherwise all plots will be created with the same
    \code{ylim} values.}

  \item{gap}{The number of lines to leave between plots.  This need not
    be an integer.}
  
  \item{...}{Additional arguments passed to
    \code{\link[Boom]{PlotDynamicDistribution}}.}

}

\seealso{
  \code{\link{plot.bsts}}
}
