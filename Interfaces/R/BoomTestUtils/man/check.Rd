\name{check}
\Rdversion{1.0}
\alias{CheckMcmcMatrix}
\alias{CheckMcmcVector}
\title{
 Check MCMC Output
}
\description{
  Verify that MCMC output covers expected values.
}
\usage{
CheckMcmcMatrix(draws, truth, confidence = .95,
                control.multiple.comparisons = TRUE)

CheckMcmcVector(draws, truth, confidence = .95)

}

\arguments{

  \item{draws}{The array of MCMC draws to check.  This must be a matrix
  for CheckMcmcMatrix and a vector for CheckMcmcVector.  }

  \item{truth}{ The vector of true values that must be covered by
    \code{draws} in order for the check to succeed.}

  \item{confidence}{ Specifies the probability width of the intervals
    used to determine whether \code{draws} covers \code{truth}.
    Central intervals are used, not HPD intervals.
  }

  \item{control.multiple.comparisons}{ If FALSE then every interval must
    cover its corresponding true value.  Otherwise a fraction of
    intervals (given by \code{confidence}) must cover.}


}
\value{

  Returns TRUE if the check passes, and FALSE if it does not.

}

\details{

  \code{CheckMcmcVector} checks a vector of draws corresponding to a
  scalar random variable.  \code{CheckMcmcMatrix} checks a matrix of
  draws corresponding to a vector of random variables.  In either case
  the check is made by constructing a central confidence interval
  (obtained by removing half of \code{1 - confidence} from the upper and
  lower tails of the distribution).

  If a single variable is being checked with CheckMcmcVector then the
  check passes if and only if the interval covers the true value.

  If multiple values are being checked with CheckMcmcMatrix then the
  user has control over how strict to make the check.  If
  \code{control.multiple.comparisons} is FALSE then the check passes if
  and only if all intervals cover true values.  Otherwise a fraction of
  intervals must cover.  The fraction is the lower bound of the binomial
  confidence interval for the coverage rate under the hypothesis that
  the true coverage rate is \code{confidence}.
  
}


\author{
  Steven L. Scott
}

\examples{

ndraws <- 100
draws <- rnorm(ndraws, 0, 1)
CheckMcmcVector(draws, 0)  ## Returns TRUE
CheckMcmcVector(draws, 17)  ## Returns FALSE

draws <- matrix(nrow = ndraws, ncol = 5)
for (i in 1:5) {
  draws[, i] <- rnorm(ndraws, i, 1)
}  
CheckMcmcMatrix(draws, truth = 1:5)  ## Returns TRUE
CheckMcmcMatrix(draws, truth = 5:1)  ## Returns FALSE

}
