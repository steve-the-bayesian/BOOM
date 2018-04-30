\name{suggest.burn.log.likelihood}
\title{Suggest MCMC Burn-in from Log Likelihood}
\alias{SuggestBurnLogLikelihood}

\description{ Suggests a burn-in period for an MCMC chain based on the
  log likelihood values simulated in the final few iterations of the
  chain.  }

\usage{
  SuggestBurnLogLikelihood(log.likelihood, fraction = .10, quantile = .9)
}

\arguments{

  \item{log.likelihood}{A numeric vector giving the log likelihood
    values for each MCMC iteration.}

  \item{fraction}{The fraction of the chain that should be used to
    determine the log likelihood lower bound.  The default setting looks
    in the final 25\% of the MCMC run.  Must be an number less than 1.
    If \code{fraction <= 0} then a 0 burn-in is returned.}

  \item{quantile}{The quantile of the values in the final fraction that
    must be exceeded before the burn-in period is declared over.}

}

\value{Returns a suggested number of iterations to discard.  This can be
  0 if \code{fraction == 0}, which is viewed as a signal that no burn-in
  is desired.}

\details{ Looks at the last \code{fraction} of the \code{log.likelihood}
  sequence and finds a specified quantile to use as a threshold.
  Returns the first iteration where \code{log.likelihood} exceeds this
  threshold.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
