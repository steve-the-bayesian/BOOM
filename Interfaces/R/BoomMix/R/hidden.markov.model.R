# Copyright 2012 Google Inc. All Rights Reserved.
# Author: stevescott@google.com (Steve Scott)

HiddenMarkovModel <- function(mixture.components,
                              state.space.size,
                              markov.model.prior = NULL,
                              niter,
                              ping = niter / 10,
                              seed = NULL) {
  ## Fits a hidden Markov model using Markov chain Monte Carlo as
  ## described in Scott (2002).
  ## Args:
  ##   mixture.components: Either a single object inheriting from
  ##     class 'MixtureComponent', or else a list of such objects.
  ##     See the help file for BoomMix-package for more information on
  ##     how to specify mixture components.
  ##   state.space.size: The number of states in the hidden Markov
  ##     chain.  If markov.model.prior is specified then this argument
  ##     is ignored and its value is inferred from markov.model.prior.
  ##   markov.model.prior: An object of class 'MarkovPrior' giving the
  ##     prior distribution of the hidden Markov chain.
  ##   niter:  The number of MCMC iterations to run.
  ##   ping: The frequency of status updates.  E.g. setting ping = 10
  ##     will print a status update every 10 MCMC iterations.
  ##   seed: An optional integer to be used for setting the C++ random
  ##     seed.  If omitted the seed will be set from the system clock.
  ##
  ## Returns:
  ##   An object of class HiddenMarkovModel, which is a list
  ##   containing the following elements:
  ##   * MCMC draws for the parameters of all the mixture
  ##     components and the hidden Markov chain.
  ##   * Log-likelihood and log-prior densities associated with each
  ##     MCMC draw.
  ##   * Marginal state membership probabilities for each data point
  ##     used to fit the model.  If 'group-id' was supplied to the
  ##     mixture components then this is a list of matrices, otherwise
  ##     it is a single matrix.  In either case, each row of a matrix
  ##     corresponds to a data point, and each column to a latent
  ##     state.
  if (inherits(mixture.components, "MixtureComponent")){
    mixture.components <- list(mixture.components)
  }

  stopifnot(is.list(mixture.components))
  stopifnot(all(sapply(mixture.components, inherits, "MixtureComponent")))
  mixture.components <- InferMixtureComponentNames(mixture.components)

  if (is.null(markov.model.prior)){
    if (missing(state.space.size)){
      stop("Error in HiddenMarkovModel:  Either 'state.space.size'",
           "or 'markov.model.prior' must be specified.")
    }
    stopifnot(state.space.size > 0)
    markov.model.prior <- MarkovPrior(state.space.size = state.space.size)
  }
  stopifnot(inherits(markov.model.prior, "MarkovPrior"))

  ans <- .Call("composite_hmm_wrapper_",
               mixture.components,
               markov.model.prior,
               niter,
               ping,
               seed)

  if (is.list(ans$state.probabilities)) {
    names(ans$state.probabilities) <- names(mixture.components[[1]]$data)
  }

  class(ans) <- "HiddenMarkovModel"
  return(ans)
}