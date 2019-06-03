% Copyright 2019 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{diff.double.model}
\title{DiffDoubleModel}
\alias{DiffDoubleModel}

\description{A 'DiffDoubleModel' is tag given to a probability
  distribution that measures a real-valued scalar outcome, and whose log
  density is twice differentiable with respect to the random variable.
  The tag is a signal to underlying C++ code that the object being
  passed is one of a subset of understood distributions.  Presently
  that subset includes the following distributions.

  \itemize{
    \item{\code{\link{SdPrior}}}
    \item{\code{\link{NormalPrior}}}
    \item{\code{\link{BetaPrior}}}
    \item{\code{\link{UniformPrior}}}
    \item{\code{\link{GammaPrior}}}
    \item{\code{\link{TruncatedGammaPrior}}}
    \item{\code{\link{LognormalPrior}}}
  }

  Clearly this list is non-exhaustive, and other distributions may be
  added in the future.

}


\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
