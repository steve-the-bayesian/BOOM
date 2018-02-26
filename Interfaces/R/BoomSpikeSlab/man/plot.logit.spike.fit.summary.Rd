\name{plot.logit.spike.fit.summary}
\Rdversion{1.1}
\alias{PlotLogitSpikeFitSummary}
\alias{PlotProbitSpikeFitSummary}
\title{
  Plot Logit or Probit Fit Summary
}
\description{
  Two plots can be accessed by this function.  The first is a time
  series plot of the "deviance R-square" statistic, by MCMC iteration.
  The second is a Hosmer-Lemeshow plot in which the data is divided into
  10 groups based on predicted probabilities, and the empirical success
  probabilities for that group are plotted against the expected
  probabilities from the model.
}

\usage{
PlotLogitSpikeFitSummary(
    model,
    burn = 0,
    which.summary = c("both", "r2", "bucket"),
    scale = c("logit", "probability"),
    cutpoint.basis = c("sample.size", "equal.range"),
    number.of.buckets = 10,
    ...)

PlotProbitSpikeFitSummary(
    model,
    burn = 0,
    which.summary = c("both", "r2", "bucket"),
    scale = c("probit", "probability"),
    cutpoint.basis = c("sample.size", "equal.range"),
    number.of.buckets = 10,
    ...)
}
\arguments{

  \item{model}{A model object inheriting from \code{\link{logit.spike}}
    or \code{\link{probit.spike}}.}

  \item{burn}{ The number of MCMC iterations in the object to be
    discarded as burn-in.  Note that this only affects the deviance
    R-square plot.  The fit summaries in the Hosmer-Lemeshow plot are
    constructed by \code{\link{logit.spike}} or
    \code{\link{probit.spike}} in order to keep permanent object sizes
    small.  }

  \item{which.summary}{Which plot is desired?}

  \item{scale}{The scale to use for the predicted probabilities in the
    Hosmer-Lemeshow plot.  }

  \item{cutpoint.basis}{How should cutpoints be determined for the
    Hosmer-Lemeshow plot?  If \code{"sample.size"} then each bucket will
    have equal sample size.  If \code{"equal.range"} then each bucket
    will occupy the same size on the chosen (logit/probit or probability) scale.}

  \item{number.of.buckets}{The number of buckets to use in the
    Hosmer-Lemeshow plot.}

  \item{\dots}{
    Additional arguments to be passed to \code{\link{barplot}.}
  }

}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{summary.lm.spike}}
  \code{\link{predict.lm.spike}}
}

\examples{
simulate.logit.spike <- function(n = 100, p = 10, ngood = 3,
                              niter=1000){
  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  prob <- plogis(x \%*\% beta)
  y <- runif(n) < prob
  x <- x[,-1]
  draws <- logit.spike(y ~ x, niter=niter)
  plot.ts(draws$beta)
  return(invisible(draws))
}
model <- simulate.logit.spike()
plot(model, "fit")
plot(model, "fit", scale = "probability", number.of.buckets = 15)
}
\keyword{models}
\keyword{regression}
