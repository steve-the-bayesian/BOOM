# Copyright 2021 Steven L. Scott. All Rights Reserved.
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA


NestedHmmPrior <- function(S0, S1, S2,
                           N0 = NULL, N1 = NULL, N2 = NULL,
                           n0 = NULL, n1 = NULL, n2 = NULL,
                           a0 = 1, a1 = 1, a2 = 1) {
  ## The prior distribution for the transition probabilities and
  ## initial distributions for the different components of the nested
  ## hmm.
  ##
  ## Args:
  ##   S0: The number of factor levels in the observed data (including
  ##     the end of session indicator, which might be implicit in the
  ##     observed data).  Can be omitted if N0 is specified.
  ##   S1: The dimension of the latent Markov chain operating at the
  ##     event-by-event level.  Can be omitted if N1 is specified.
  ##   S2: The dimension of the latent Markov chain operating at the
  ##     session-by-session level.  Can be omitted if N2 is specified.
  ##   N0: An S0 x S0 matrix of prior transition counts for the
  ##     observed data.
  ##   n0: An S0 vector of prior observations for the observed data.
  ##   N1: An S1 x S1 matrix of prior transition counts for the
  ##     latent event-level chain.
  ##   n1: An S1 vector of prior observations for the latent
  ##     event-level chain.
  ##   N2: An S2 x S2 matrix of prior transition counts for the
  ##     latent session-level chain.
  ##   n2: An S2 vector of prior observations for the latent
  ##     session-level chain.
  ##   a0: If N0 or n0 are omitted, they will be replace by a constant
  ##     matrix/vector with this value.
  ##   a1: If N1 or n1 are omitted, they will be replace by a constant
  ##     matrix/vector with this value.
  ##   a2: If N2 or n2 are omitted, they will be replace by a constant
  ##     matrix/vector with this value.
  ##
  ## Returns:
  ##   An object of class NestedHmmPrior, which is a list with the
  ##   elements N0, n0, N1, n1, N2, and n2.
  if (is.null(N0)) {
    N0 <- matrix(a0, nrow = S0, ncol = S0)
  }
  stopifnot(is.matrix(N0))
  stopifnot(all(N0>=0))
  stopifnot(nrow(N0) == ncol(N0))

  if (is.null(n0)) {
    n0 <- rep(a0, S0)
  }
  stopifnot(is.numeric(n0))
  stopifnot(length(n0) == nrow(N0))
  stopifnot(all(n0 >= 0))

  if (is.null(N1)) {
    N1 <- matrix(a1, nrow = S1, ncol = S1)
  }
  stopifnot(is.matrix(N1))
  stopifnot(all(N1 >= 0))
  stopifnot(nrow(N1) == ncol(N1))

  if (is.null(n1)) {
    n1 <- rep(a1, S1)
  }
  stopifnot(is.numeric(n1))
  stopifnot(length(n1) == nrow(N1))
  stopifnot(all(n1 >= 0))

  if (is.null(N2)) {
    N2 <- matrix(a2, nrow = S2, ncol = S2)
  }
  stopifnot(is.matrix(N2))
  stopifnot(all(N2 >= 0))
  stopifnot(nrow(N2) == ncol(N2))

  if (is.null(n2)) {
    n2 <- rep(a2, S2)
  }
  stopifnot(is.numeric(n2))
  stopifnot(length(n2) == nrow(N2))
  stopifnot(all(n2 >= 0))

  ans <- list(N0 = N0,
              N1 = N1,
              N2 = N2,
              n0 = n0,
              n1 = n1,
              n2 = n2)

  class(ans) <- "NestedHmmPrior"
  return(ans)
}

##======================================================================
NestedHmm <- function(streams,
                      nested.hmm.prior,
                      eos = "zzzEND",
                      niter,
                      burn = niter / 10,
                      ping = niter / 10,
                      threads = 1,
                      seed = NULL,
                      print.sufficient.statistics = 3) {
  ## Fits a nested hidden Markov model using Markov chain Monte Carlo.
  ## Args:
  ##   streams: A list of objects inheriting from NestedHmmStream,
  ##     each describing a stream of observations to be modeled.
  ##   nested.hmm.prior:  A prior of class NestedHmmPrior.
  ##   eos: The string used to label the end of session factor level.
  ##     The end of session indicator may be explicit or implicit.  If
  ##     explicit, the last level of the factor must match eos.  If
  ##     implicit, the eos indicator will be added to each session
  ##     automatically.
  ##   niter:  An integer giving the desired number of MCMC iterations.
  ##   ping: The frequency with which status reports should be
  ##     printed.  ping = 10 will print a status update every 10
  ##     iterations.
  ##   threads: The number of CPU threads to use in the MCMC.  If
  ##     'threads' is greater than the number of cores on the machine,
  ##     the number on the machine will be used.
  ##   seed: The seed to use for the C++ random number generator.  If
  ##     NULL then the RNG will be seeded from the time stamp.
  ##   print.sufficient.statistics: For debugging purposes, the model
  ##     can print the sufficient statistics at each iteration.  If
  ##     print.sufficient.statistics is 2 or less then the session
  ##     level (level 2) sufficient statistics will be printed.  If 1
  ##     or less then the latent-event-type sufficient statistics will
  ##     be printed.  If 0 or less then all will be printed.
  if (is.factor(streams[[1]])) {
    streams <- list(streams)
  }
  stopifnot(is.list(streams))
  ## streams[[1]] is the first stream, which is a list of sessions.
  stopifnot(is.list(streams[[1]]))

  ## streams[[1]][[1]] is the first session, which is a factor.
  stopifnot(is.factor(streams[[1]][[1]]))

  ## Check that the eos indicator is either left out, or is in the
  ## right place.
  stopifnot(is.character(eos))
  factor.levels <- levels(streams[[1]][[1]])
  if (eos %in% factor.levels && tail(factor.levels, 1) != eos) {
    stop("If eos is explicitly present, it must be the last level ",
         "of the factor.  To adjust the levels of a factor, try ",
         "'x <- factor(x, levels = newlevels)")
  }

  stopifnot(inherits(nested.hmm.prior, "NestedHmmPrior"))

  if (!is.null(seed)) {
    seed <- as.integer(seed)
  }

  ans <- .Call("nested_hmm_wrapper_",
               streams,
               as.character(eos),
               nested.hmm.prior,
               as.integer(niter),
               as.integer(burn),
               as.integer(ping),
               as.integer(threads),
               seed,
               as.integer(print.sufficient.statistics))
  class(ans) <- "NestedHmm"

  if (tail(factor.levels, 1) != eos) {
    factor.levels <- c(factor.levels, eos)
  }

  dimnames(ans$observed.data.initial.distributions) <-
    list(iteration = NULL,
         latent.session = NULL,
         latent.event = NULL,
         observed.event = factor.levels)

  dimnames(ans$observed.data.transition.probabilities) <-
    list(iteration = NULL,
         latent.session = NULL,
         latent.event = NULL,
         observed.event.origin = factor.levels,
         observed.event.destination = factor.levels)

  return(ans)
}

## TODO(stevescott):
## Add plot and summary functions.
