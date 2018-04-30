% Copyright 2011 Google Inc. All Rights Reserved.
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

\name{plot.bsts}
\title{Plotting functions for Bayesian structural time series}

\alias{plot.bsts}
\alias{PlotBstsCoefficients}
\alias{PlotBstsComponents}
\alias{PlotBstsState}
\alias{PlotBstsResiduals}
\alias{PlotBstsPredictionErrors}
\alias{PlotBstsSize}
\alias{PlotDynamicRegression}
\alias{PlotBstsForecastDistribution}
\alias{PlotSeasonalEffect}
\alias{PlotMonthlyAnnualCycle}

\description{Functions to plot the results of a model fit using
  \code{\link{bsts}}.}

\usage{
  \method{plot}{bsts}(x, y = c("state", "components", "residuals",
              "coefficients", "prediction.errors",
              "forecast.distribution",
              "predictors", "size", "dynamic", "seasonal", "monthly",
              "help"),
      ...)

  PlotBstsCoefficients(bsts.object, burn = SuggestBurn(.1, bsts.object),
                       inclusion.threshold = 0, number.of.variables = NULL, ...)

  PlotBstsComponents(bsts.object,
                     burn = SuggestBurn(.1, bsts.object),
                     time,
                     same.scale = TRUE,
                     layout = c("square", "horizontal", "vertical"),
                     style = c("dynamic", "boxplot"),
                     ylim = NULL,
                     components = 1:length(bsts.object$state.specification),
                     ...)

  PlotDynamicRegression(bsts.object,
                        burn = SuggestBurn(.1, bsts.object),
                        time = NULL,
                        same.scale = FALSE,
                        style = c("dynamic", "boxplot"),
                        layout = c("square", "horizontal", "vertical"),
                        ylim = NULL,
                        zero.width = 2,
                        zero.color = "green",
                        ...)

  PlotBstsState(bsts.object, burn = SuggestBurn(.1, bsts.object),
                       time, show.actuals = TRUE,
                       style = c("dynamic", "boxplot"),
                       scale = c("linear", "mean"),
                       ylim = NULL,
                       ...)

  PlotBstsResiduals(bsts.object, burn = SuggestBurn(.1, bsts.object),
                       time, style = c("dynamic", "boxplot"), means =
                       TRUE, ...)

  PlotBstsPredictionErrors(bsts.object, cutpoints = NULL,
                           burn = SuggestBurn(.1, bsts.object),
                           style = c("dynamic", "boxplot"),
                           xlab = "Time", ylab = "", main = "",
                           ...)

  PlotBstsForecastDistribution(bsts.object, cutpoints = NULL,
                               burn = SuggestBurn(.1, bsts.object),
                               style = c("dynamic", "boxplot"),
                               xlab = "Time",
                               ylab = "",
                               main = "",
                               show.actuals = TRUE,
                               col.actuals = "blue",
                               ...)

  PlotBstsSize(bsts.object, burn = SuggestBurn(.1, bsts.object), style =
                       c("histogram", "ts"), ...)

  PlotSeasonalEffect(bsts.object, nseasons = 7, season.duration = 1,
                     same.scale = TRUE, ylim = NULL, get.season.name = NULL,
                     burn = SuggestBurn(.1, bsts.object),  ...)

  PlotMonthlyAnnualCycle(bsts.object, ylim = NULL, same.scale = TRUE,
                     burn = SuggestBurn(.1, bsts.object),  ...)
}

\arguments{
  \item{x}{An object of class \code{\link{bsts}}.}

  \item{bsts.object}{An object of class \code{\link{bsts}}.}

  \item{y}{A character string indicating the aspect of the model that
    should be plotted.}

  \item{burn}{The number of MCMC iterations to discard as burn-in.}

  \item{col.actuals}{The color to use for the actual data when comparing
    actuals vs forecasts.}

  \item{components}{A numeric vector indicating which components to
    plot. Component indices correspond to elements of the state
    specification that was used to build the bsts model being plotted.}

  \item{cutpoints}{A numeric vector of integers, or \code{NULL}.  For
    diagnostic plots of prediction errors or forecast distributions, the
    model will be re-fit with a separate MCMC run for each entry in
    'cutpoints'.  Data up to each cutpoint will be included in the fit,
    and one-step prediction errors for data after the cutpoint will be
    computed.
  }

  \item{get.season.name}{A function that can be used to infer the title
    of each seasonal plot.  It should take a single \code{\link{POSIXt}},
    \code{\link{Date}}, or similar object as an argument, and return
    a single string that can be used as a panel title.  If
    \code{get.season.name} is \code{NULL} and \code{nseasons} is
    specified or inferred to be one of the following values, then the
    following functions will be used.
    \itemize{
      \item 4:  \code{\link{quarters}}
      \item 7:  \code{\link{weekdays}}
      \item 12: \code{\link{months}}
    }
  }

  \item{inclusion.threshold}{An inclusion probability that individual
    coefficients must exceed in order to be displayed when \code{what ==
    "coefficients"}.  See the help file for
    \code{\link[BoomSpikeSlab]{plot.lm.spike}.}}

  \item{layout}{For controlling the layout of functions that generate
    mutiple plots.}

  \item{main}{Main title for the plot.}

  \item{means}{Logical.  If TRUE then the mean of each residual is
    plotted as a blue dot.  If false only the distribution of the
    residuals is plotted.}
  
  \item{nseasons}{If there is only one seasonal component in the model,
    this argument is ignored.  If there are multiple seasonal
    components then \code{nseasons} and \code{season.duration} are used
    to select the desired one.}

  \item{number.of.variables}{If non-\code{NULL} this specifies the
    number of coefficients to plot, taking precedence over
    \code{inclusion.threshold}.  See \code{\link[BoomSpikeSlab]{plot.lm.spike}.}}

  \item{same.scale}{Logical.  If \code{TRUE} then all the state
    components will be plotted with the same scale on the vertical axis.
    If \code{FALSE} then each component will get its own scale for the
    vertical axis.}

  \item{scale}{The scale on which to plot the state.  If the error
    family is "logit" or "poisson" then the state can either be plotted
    on the scale of the linear predictor (e.g. trend + seasonal +
    regression) or the linear predictor can be passed through the link
    function so as to plot the distribution of the conditional mean.}

  \item{season.duration}{If there is only one seasonal component in the
    model, this argument is ignored.  If there are multiple seasonal
    components then \code{nseasons} and \code{season.duration} are used
    to select the desired one.}

  \item{show.actuals}{Logical.  If \code{TRUE} then actual values from
    the fitted series will be shown on the plot.}

  \item{style}{The desired plot style.  Partial matching is allowed, so
    "dyn" would match "dynamic", for example.}

  \item{time}{An optional vector of values to plot against.  If missing,
    the default is to diagnose the time scale of the original time
    series.}

  \item{xlab}{Label for the horizontal axis.}

  \item{ylab}{Label for the vertical axis.}

  \item{ylim}{Limits for the vertical axis.  If \code{NULL} these will
    be inferred from the state components and the \code{same.scale}
    argument.  Otherwise all plots will be created with the same
    \code{ylim} values.}

  \item{zero.width}{A numerical value for the width of the reference
    line at zero.  If \code{NULL} then the line will be omitted.}

  \item{zero.color}{A color for the width of the reference line at zero.
    If \code{NULL} then the line will be omitted.}
  
  \item{...}{Additional arguments to be passed to
    \code{\link[Boom]{PlotDynamicDistribution}}, or
    \code{\link[Boom]{TimeSeriesBoxplot}}.}

}

\details{

  \code{\link{PlotBstsState}}, \code{\link{PlotBstsComponents}}, and
  \code{\link{PlotBstsResiduals}} all produce dynamic distribution
  plots.  \code{\link{PlotBstsState}} plots the aggregate state
  contribution (including regression effects) to the mean, while
  \code{\link{PlotBstsComponents}} plots the contribution of each state
  component.  \code{\link{PlotBstsResiduals}} plots the posterior
  distribution of the residuals given complete data (i.e. looking
  forward and backward in time).  \code{\link{PlotBstsPredictionErrors}}
  plots filtering errors (i.e. the one-step-ahead prediction errors
  given data up to the previous time point).
  \code{\link{PlotBstsForecastDistribution}} plots the one-step-ahead
  forecasts instead of the prediction errors.

  \code{\link{PlotBstsCoefficients}} creates a significance plot for
  the predictors used in the state space regression model.  It is
  obviously not useful for models with no regressors.

  \code{\link{PlotBstsSize}} plots the distribution of the number of
  predictors included in the model.

  \code{\link{PlotSeasonalEffect}} generates an array of plots showing
  how the distibution of the seasonal effect changes, for each season,
  for models that include a seasonal state component.

  \code{\link{PlotMonthlyAnnualCycle}} produces an array of plots much
  like \code{\link{PlotSeasonalEffect}}, for models that include a
  \code{\link{MonthlyAnnualCycle}} state component.
}

\value{
  These functions are called for their side effect, which is to produce
  a plot on the current graphics device.

  \code{PlotBstsState} invisibly returns the state object being plotted.
}

\examples{
  data(AirPassengers)
  y <- log(AirPassengers)
  ss <- AddLocalLinearTrend(list(), y)
  ss <- AddSeasonal(ss, y, nseasons = 12)
  model <- bsts(y, state.specification = ss, niter = 500)
  plot(model, burn = 100)
  plot(model, "residuals", burn = 100)
  plot(model, "components", burn = 100)
  plot(model, "forecast.distribution", burn = 100)
}

\seealso{
  \code{\link{bsts}}
  \code{\link[Boom]{PlotDynamicDistribution}}
  \code{\link[BoomSpikeSlab]{plot.lm.spike}}
}
