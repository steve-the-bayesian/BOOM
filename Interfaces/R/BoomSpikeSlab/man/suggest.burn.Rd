% Copyright 2018 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{suggest.burn}
\Rdversion{1.1}
\alias{SuggestBurn}
\title{
  Suggest Burn-in
}
\description{
  Suggest a burn-in period for a Bayesian neural network model.
}
\usage{
SuggestBurn(model)
}

\arguments{
  \item{model}{An object inheriting from class \code{BayesNnet}. }
}

\details{

  See \code{\link{SuggestBurnLogLikelihood}} for details of the on how
  the burn-in period is suggested.  In this case the negative the
  residual standard deviation is used as a proxy for log likelihood.
  
}

\value{
  A non-negative integer less than the number of MCMC draws.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link[Boom]{SuggestBurnLogLikelihood}}
}

