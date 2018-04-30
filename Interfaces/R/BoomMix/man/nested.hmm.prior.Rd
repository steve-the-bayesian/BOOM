\alias{NestedHmmPrior}
\name{nested.hmm.prior}
\title{Prior for nested hidden Markov model}
\Rdversion{1.0}

\description{
  For specifying the prior distribution of a nested HMM.
}

\usage{
NestedHmmPrior(S0, S1, S2,
               N0 = NULL, N1 = NULL, N2 = NULL,
               n0 = NULL, n1 = NULL, n2 = NULL,
               a0 = 1, a1 = 1, a2 = 1)
}

\arguments{
  \item{S0}{Dimension of the raw data (including the end of session
    indicator).  Can be omitted if \code{N0} is supplied.}

  \item{S1}{Dimension of the event-level latent Markov chain.  Can be
    omitted if \code{N1} is supplied.}

  \item{S2}{Dimension of the session-level latent Markov chain.  Can be
    omitted if \code{N2} is supplied.}

  \item{N0}{An \code{S0} by \code{S0} matrix of prior counts of observed
    data transitions.  In this and all other transition matrices,
    transitions are from rows to columns.  The same prior counts are
    used on all event-level chains.}

  \item{n0}{An {S0} vector of prior counts for the initial distributions
    of observed data.  The last element should be zero, because the
    process cannot start on the end-of-session indicator.}

  \item{N1}{An \code{S1} by \code{S1} matrix of prior transition counts
    for the event-level latent Markov chains.  The same prior counts are
    used on all event-level chains.}

  \item{n1}{An {S1} vector of prior counts for the initial distributions
    of the latent event-level Markov chains.}

  \item{N2}{An \code{S2} by \code{S2} matrix of prior transition counts
    for the session-level latent Markov chains.  }

  \item{n2}{An {S2} vector of prior counts for the initial distributions
    of the latent session-level Markov chains.}

  \item{a0}{If \code{N0} or \code{n0} are missing then they will be
    replaced by an appropriately sized constant matrix with elements
    \code{a0}.}

  \item{a1}{If \code{N1} or \code{n1} are missing then they will be
    replaced by an appropriately sized constant matrix with elements
    \code{a1}.}

  \item{a2}{If \code{N2} or \code{n2} are missing then they will be
    replaced by an appropriately sized constant matrix with elements
    \code{a2}.}

}

\value{An object of class NestedHmmPrior, which is a list containing
  \code{N0}, \code{N1}, \code{N2}, \code{n0}, \code{n1}, and \code{n2}.
}


\references{

  Scott (2006), "A Nested Hidden Markov Model for Internet Browsing Behavior".
  \url{http://stevethebayesian.com/nested-hmm.pdf}
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{NestedHmm}}
}

\keyword{models}
