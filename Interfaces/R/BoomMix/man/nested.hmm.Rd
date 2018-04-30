\alias{NestedHmm}
\name{nested.hmm}
\title{Nested hidden Markov models}
\Rdversion{1.0}

\description{
  Fit a nested hidden Markov model using MCMC.
}

\usage{
NestedHmm(streams,
          nested.hmm.prior,
          eos = "zzzEND",
          niter,
          burn = niter / 10,
          ping = niter / 10,
          threads = 1,
          seed = NULL,
          print.sufficient.statistics = 3)
}

\arguments{
  \item{streams}{A list of objects inheriting from NestedHmmStream,
    each describing a stream of observations to be modeled.}

  \item{nested.hmm.prior}{A prior of class \code{\link{NestedHmmPrior}}.}

  \item{eos}{ The string used to label the end of session factor level.
    The end of session indicator may be explicit or implicit.  If
    explicit, the last level of the factor must match eos.  If
    implicit, the eos indicator will be added to each session
    automatically.}

  \item{niter}{An integer giving the desired number of MCMC iterations,
    after burn-in.}

  \item{burn}{The desired number of burn-in iterations.}

  \item{ping}{The frequency with which status reports should be
    printed.  ping = 10 will print a status update every 10
    iterations.}

  \item{threads}{The number of CPU threads to use in the MCMC.  If
    'threads' is greater than the number of cores on the machine,
    the number on the machine will be used.}

  \item{seed}{The seed to use for the C++ random number generator.  If
    NULL then the RNG will be seeded from the time stamp.}

  \item{print.sufficient.statistics}{For debugging purposes, the model
    can print the sufficient statistics at each iteration.  If
    print.sufficient.statistics is 2 or less then the session
    level (level 2) sufficient statistics will be printed.  If 1
    or less then the latent-event-type sufficient statistics will
    be printed.  If 0 or less then all will be printed.}

}

\value{

  An object of class \code{NestedHmm}, which is a list with the
  following elements.  For all objects representing Monte Carlo draws,
  the leading index is the iteration number.  \code{S0} refers to the
  number of levels in the observed data (including the end of session
  indicator).  \code{S1} is the dimension of the event-level latent
  Markov chain.  \code{S2} is the dimension of the session-level chain.

  \item{latent.session.initial.distribution}{
    A \code{niter} by \code{S2} matrix of MCMC draws for the initial
    distribution of the session level chain.
  }

  \item{latent.session.transition.probabilities}{A \code{niter} by
    \code{S2} by \code{S2} array of MCMC draws for the transition
    probability matrix of the session level latent Markov chain.
    \code{latent.session.transition.probabilities[i, r, s]} is the
    probability of a transition to state \code{s}, given the previous
    session was in state \code{r}.  }

  \item{latent.event.initial.distributions}{A \code{niter} by \code{S2}
    by \code{S1} array of Monte Carlo draws for the initial
    distributions of the event-level latent Markov chains.
    \code{latent.event.initial.distributions[i, H, h]} is the
    probability that the event level initial distribution starts in
    state \code{h} given the session level chain that it belongs to is
    in state \code{H}. }

  \item{latent.event.transition.probabilities}{A niter by S2 by S1 by S1
    array of Monte Carlo draws for the transition probability matrices
    of the event-level latent Markov chains.
    \code{latent.event.transition.probabilities[i, H, h0, h1]} is the
    probability that the event level chain transitions to state
    \code{h1}, given that it started in state \code{h0}, and the session level
    chain that it belongs to is in state \code{H}.  }


  \item{observed.data.initial.distributions}{A \code{niter} by \code{S2}
    by \code{S1} by \code{S0} array of Monte Carlo draws for the initial
    distributions of the data-level latent Markov chains.
    \code{observed.data.initial.distributions[i, H, h, s]} is the
    probability that the data-level chain starts in state \code{s} given
    the session level chain that it belongs to is in state \code{H} and
    the event-level chain is in state \code{h}.  }

  \item{observed.data.transition.probabilities}{A \code{niter} by
    \code{S2} by \code{S1} by \code{S0} by \code{S0} array of Monte
    Carlo draws for the transition probabilities of the observed data
    chains.  \code{observed.data.transition.probabilities[i, H, h, r,
    s]} is the probability the observed data transition from level r to
    level s given the session level chain that they belong to is in
    state \code{H} and the event-level chain is in state \code{h}.}

  \item{log.likelihood}{The log likelihood values corresponding to each
    set of Monte Carlo draws.}

  \item{log.posterior}{The un-normalized log posterior values (log
    likelihood + log prior) corresponding to each set of Monte Carlo
    draws.}

  \item{session.type.distribution}{A list giving the marginal
    distribution of the session type for each stream that was passed to
    the model.  Each list element is a matrix corresponding to a stream.
    The rows of the matrix correspond to the sessions in the stream.
    The columns correspond to session types.  The distribution is a
    Monte Carlo estimate based on the MCMC iterations after burn-in.  It
    is not Rao-Blackwellized.  Note that this method of inference is
    vulnerable to label switching.  }

}

\references{
  Scott (2006), "A Nested Hidden Markov Model for Internet Browsing Behavior".
  \url{http://stevethebayesian.com/nested-hmm.pdf}
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{
## TODO(stevescott):  add some examples.
}
\keyword{models}
