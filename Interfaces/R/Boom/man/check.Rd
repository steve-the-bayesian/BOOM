\name{check}
\Rdversion{1.0}
\alias{CheckMcmcMatrix}
\alias{CheckMcmcVector}
\alias{McmcMatrixReport}
\title{
 Check MCMC Output
}
\description{
  Verify that MCMC output covers expected values.
}
\usage{
CheckMcmcMatrix(draws, truth, confidence = .95,
                control.multiple.comparisons = TRUE,
                burn = 0)

CheckMcmcVector(draws, truth, confidence = .95, burn = 0)

McmcMatrixReport(draws, truth, confidence = .95, burn = 0)
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

  \item{burn}{The number of MCMC iterations to discard as burn-in.}
}
\value{
  
  \code{CheckMcmcVector} and \code{CheckMcmcMatrix} return \code{TRUE}
  if the check passes, and \code{FALSE} if it does not.

  \code{McmcMatrixReport} returns a string that can be put in the
  \code{info} field of an \code{\link[testthat]{expect_true}}
  expression, to give useful information about a failed test case.  The
  return value is a textual representation of a three column matrix.
  Each row matches a variable in \code{draws}, and gives the lower and
  upper bounds for the credible interval used to check the values.  The
  final column lists the true values that are supposed to be inside the
  credible intervals.  The value is returned as a character string that
  is expected to be fed to \code{cat()} or \code{print()} so that it
  will render correctly in R CMD CHECK output.
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
