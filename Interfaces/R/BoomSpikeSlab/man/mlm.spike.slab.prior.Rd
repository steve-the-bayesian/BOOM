\name{mlm.spike.slab.prior}
\Rdversion{1.1}
\alias{MultinomialLogitSpikeSlabPrior}
\title{
  Create a spike and slab prior for use with mlm.spike.
}
\description{
  Creates a spike and slab prior for use with mlm.spike.
}
\usage{
MultinomialLogitSpikeSlabPrior(
    response,
    subject.x,
    expected.subject.model.size = 1,
    choice.x = NULL,
    expected.choice.model.size = 1,
    max.flips = -1,
    nchoices = length(levels(response)),
    subject.dim = ifelse(is.null(subject.x), 0, ncol(subject.x)),
    choice.dim = ifelse(is.null(choice.x), 0, ncol(choice.x)))
}
\arguments{
\item{response}{The response variable in the multinomial logistic
  regression.  The response variable is optional if nchoices
  is supplied.  If 'response' is provided then the prior
  means for the subject level intercpets will be chosen to
  match the empirical values of the response.}

\item{subject.x}{The design matrix for subject-level predictors.
  This can be NULL or of length 0 if no subject-level
  predictors are present.}

\item{expected.subject.model.size}{The expected number of non-zero
  coefficients -- per choice level -- in the subject specific
  portion of the model.  All coefficients can be forced into
  the model by setting this to a negative number, or by setting
  it to be larger than the dimension of the subject-level
  predictors.}

\item{choice.x}{The design matrix for choice-level predictors.  Each
  row of this matrix represents the characteristics of a choice
  in a choice occasion, so it takes 'nchoices' rows to encode
  one observation.  This can be NULL or of length 0 if no
  choice-level predictors are present.}

\item{expected.choice.model.size}{The expected number of non-zero
  coefficients in the choice-specific portion of the model.
  All choice coefficients can be forced into the model by
  setting this to a negative number, or by setting it to be
  larger than the dimension of the choice-level predictors (for
  a single response level).}

\item{max.flips}{The maximum number of variable inclusion indicators
    the sampler will attempt to sample each iteration.  If
    \code{max.flips <= 0} then all indicators will be sampled.}

\item{nchoices}{Tne number of potential response levels.}

\item{subject.dim}{The number of potential predictors in the
  subject-specific portion of the model.}

\item{choice.dim}{The number of potential predictors in the
  choice-specific portion of the model.}
}

\value{ An object of class \code{\link{IndependentSpikeSlabPrior}}, with
  elements arranged as expected by \code{\link{mlm.spike}}.
}
\author{
  Steven L. Scott
}

\references{
  Tuchler (2008), "Bayesian Variable Selection for Logistic Models Using
  Auxiliary Mixture Sampling", \emph{Journal of Computational and
  Graphical Statistics}, \bold{17} 76 -- 94.
}

\keyword{models}
\keyword{regression}
