% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{simulate.fake.mixed.frequency.data}

\alias{SimulateFakeMixedFrequencyData}

\title{Simulate fake mixed frequency data}


\Rdversion{1.0}
\description{
  Simulate a fake data set that can be used to test mixed frequency code.
}

\usage{
SimulateFakeMixedFrequencyData(nweeks,
                               xdim,
                               number.nonzero = xdim,
                               start.date = as.Date("2009-01-03"),
                               sigma.obs = 1.0,
                               sigma.slope = .5,
                               sigma.level = .5,
                               beta.sd = 10)

}

\arguments{

  \item{nweeks}{The number of weeks of data to simulate.}

  \item{xdim}{The dimension of the predictor variables to be simulated.}

  \item{number.nonzero}{The number nonzero coefficients.  Must be
    less than or equal to \code{xdim}.}

  \item{start.date}{The date of the first simulated week.}

  \item{sigma.obs}{The residual standard deviation for the fine time
    scale model.}

  \item{sigma.slope}{The standard deviation of the slope component of
    the local linear trend model for the fine time scale data.}

  \item{sigma.level}{The standard deviation of the level component fo
    the local linear trend model for the fine time scale data.}

  \item{beta.sd}{The standard deviation of the regression coefficients
    to be simulated.}

}

\value{

  Returns a list with the following components

  \item{coarse.target}{A \code{\link[zoo]{zoo}} time series containing the
    monthly values to be modeled.}

  \item{fine.target}{A \code{\link[zoo]{zoo}} time series containing the
    weekly observations that aggregate to \code{coarse.target}. }

  \item{predictors}{A \code{\link[zoo]{zoo}} matrix corresponding to
    \code{fine.target} containing the set of predictors variables to use
    in \code{\link{bsts.mixed}} prediction.}


  \item{true.beta}{The vector of "true" regression coefficients used to
    simulate \code{fine.target}.}

  \item{true.sigma.obs}{The residual standard deviation that was used to
    simulate \code{fine.target}.}

  \item{true.sigma.slope}{The value of \code{sigma.slope} used to
    simulate \code{fine.target}.}

  \item{true.sigma.level}{The value of \code{sigma.level} use to
    simulate \code{fine.target}.}

  \item{true.trend}{The combined contribution of the simulated latent
    state on \code{fine.target}, including regression effects.}

  \item{true.state}{A matrix containin the fine-scale state of the model
    being simulated.  Columns represent time (weeks).  Rows correspond
    to regression (a constant 1), the local linear trend level, the
    local linear trend slope, the values of \code{fine.target}, and the
    weekly partial aggregates of \code{coarse.target}.}

}

\details{

  The simulation begins by simulating a local linear trend model for
  \code{nweeks} to get the trend component.

  Next a \code{nweeks} by \code{xdim} matrix of predictor variables is
  simulated as IID normal(0, 1) deviates, and a \code{xdim}-vector of
  regression coefficients is simulated as IID normal(0, \code{beta.sd}).
  The product of the predictor matrix and regression coefficients is
  added to the output of the local linear trend model to get
  \code{fine.target}.

  Finally, \code{fine.target} is aggregated to the month level to get
  \code{coarse.target}.

}

\references{
  Harvey (1990), "Forecasting, structural time series, and the Kalman
  filter", Cambridge University Press.

  Durbin and Koopman (2001), "Time series analysis by state space
  methods", Oxford University Press.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{bsts.mixed}},
  \code{\link{AddLocalLinearTrend}},
}

\examples{
  fake.data <- SimulateFakeMixedFrequencyData(nweeks = 100, xdim = 10)
  plot(fake.data$coarse.target)
}

\keyword{models}
\keyword{regression}
