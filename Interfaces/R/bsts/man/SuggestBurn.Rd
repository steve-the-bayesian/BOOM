% Copyright 2011 Google Inc. All Rights Reserved.
% Author: stevescott@google.com (Steve Scott)

\name{SuggestBurn}
\title{Suggested burn-in size}

\alias{SuggestBurn}

\description{Suggest the size of an MCMC burn in sample as a proportion
  of the total run.}

\usage{SuggestBurn(proportion, bsts.object)}

\arguments{

  \item{proportion}{The proportion of the MCMC run to discard as burn in.}
  \item{bsts.object}{An object of class \code{\link{bsts}}.}
}

\value{
  An integer number of iterations to discard.
}

\seealso{
  \code{\link{bsts}}
}


