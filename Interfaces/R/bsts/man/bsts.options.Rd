\name{bsts.options.Rd}

\alias{BstsOptions}

\title{Bsts Model Options}
\Rdversion{1.0}

\description{
  Rarely used modeling options for bsts models.
}

\usage{
BstsOptions(save.state.contributions = TRUE,
            save.prediction.errors = TRUE,
            bma.method = c("SSVS", "ODA"),
            oda.options = list(
                fallback.probability = 0.0,
                eigenvalue.fudge.factor = 0.01),
            timeout.seconds = Inf)

}

\arguments{

  \item{save.state.contributions}{Logical.  If \code{TRUE} then a 3-way
    array named \code{state.contributions} will be stored in the
    returned object.  The indices correspond to MCMC iteration, state
    model number, and time.  Setting \code{save.state.contributions} to
    \code{FALSE} yields a smaller object, but \code{plot} will not be
    able to plot the the "state", "components", or "residuals" for the
    fitted model.}

  \item{save.prediction.errors}{Logical.  If \code{TRUE} then a matrix
    named \code{one.step.prediction.errors} will be saved as part of the
    model object.  The rows of the matrix represent MCMC iterations, and
    the columns represent time.  The matrix entries are the
    one-step-ahead prediction errors from the Kalman filter.  }

  \item{bma.method}{If the model contains a regression component, this
    argument specifies the method to use for Bayesian model averaging.
    "SSVS" is stochastic search variable selection, which is the classic
    approach from George and McCulloch (1997).  "ODA" is orthoganal data
    augmentation, from Ghosh and Clyde (2011).  It adds a set of latent
    observations that make the \eqn{X^TX}{X'X} matrix diagonal, vastly
    simplifying complete data MCMC for model selection.}

  \item{oda.options}{If bma.method == "ODA" then these are some options
    for fine tuning the ODA algorithm.
    \itemize{

      \item \code{fallback.probability}: Each MCMC iteration will use
      SSVS instead of ODA with this probability.  In cases where
      the latent data have high leverage, ODA mixing can suffer.
      Mixing in a few SSVS steps can help keep an errant algorithm
      on track.

      \item \code{eigenvalue.fudge.factor}: The latent X's will be
      chosen so that the complete data \eqn{X^TX}{X'X} matrix (after
      scaling) is a constant diagonal matrix equal to the largest
      eigenvalue of the observed (scaled) \eqn{X^TX}{X'X} times (1 +
      eigenvalue.fudge.factor).  This should be a small positive number.
      } }

  \item{timeout.seconds}{The number of seconds that sampler will be
    allowed to run.  If the timeout is exceeded the returned object will
    be truncated to the final draw that took place before the timeout
    occurred, as if that had been the requested number of iterations.}
}

\value{
  The arguments are checked to make sure they have legal types and
  values, then a list is returned containing the arguments.
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\keyword{chron}
