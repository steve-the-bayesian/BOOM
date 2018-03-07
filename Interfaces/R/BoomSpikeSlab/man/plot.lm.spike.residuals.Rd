\name{plot.lm.spike.residuals}
\Rdversion{1.1}
\alias{PlotLmSpikeResiduals}
\title{
  Residual plot for lm.spike
}
\description{
  Plot residuals vs. fitted values in an lm.spike model.
}

\usage{
PlotLmSpikeResiduals(
    object,
    burn = SuggestBurnLogLikelihood(object$log.likelihood),
    ...)

}
\arguments{

  \item{object}{A model object inheriting from \code{\link{lm.spike}}.}

  \item{burn}{ The number of MCMC iterations to be discarded as burn-in
    before computing posterior means. }

  \item{\dots}{
    Additional arguments passed to \code{\link{plot}.}
  }

}

\details{
  This plot is normally called via the plot function for \code{lm.spike}
  objects.  See the help entry for \code{\link{lm.spike}} for example
  usage.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{plot.lm.spike}}
}

\keyword{models}
\keyword{regression}
