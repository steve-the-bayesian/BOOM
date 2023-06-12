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


FiniteMixture <- function(mixture.component.specification,
                          state.space.size = NULL,
                          weight.prior = NULL,
                          niter,
                          ping = niter / 10,
                          known.source = NULL,
                          seed = NULL) {
  ## Args:
  ##   mixture.component.specification: A list of objects inheriting from class
  ##     MixtureComponent.  If the list would otherwise be of length 1
  ##     then this can be a single object of class MixtureComponent.
  ##     MixtureComponent objects contain their data and a prior
  ##     specification.
  ##   state.space.size: The number of states in the mixture chain.
  ##     If weight.prior is specified then this argument is ignored
  ##     and its value is inferred from weight.prior.
  ##   weight.prior: An object of class DirichletPrior.  The
  ##     dimension of this object is the number of latent states in
  ##     the finite mixture.
  ##   niter:  The desired number of MCMC iterations.
  ##   ping: The frequency of status updates during the MCMC.
  ##     E.g. setting ping = 100 will print a status update every 100
  ##     MCMC iterations.
  ##   known.source: An optional numeric vector indicating which
  ##     mixture component individual observations belong to.  In a
  ##     typical finite mixture problem this information will be
  ##     unavailable.  If it is fully available then the "finite
  ##     mixture model" turns into a naive Bayes classifier.  If the
  ##     components for only a few observations are known then the
  ##     unknown ones can be marked with NA, in which case the model
  ##     becomes a "semi-supervised learner."
  ##   seed: An integer to use as the random seed for the underlying
  ##     C++ code.  If \code{NULL} then the seed will be set using the
  ##     clock.
  ##
  ## Returns:
  ##   An object of class FiniteMixture, which is a list with
  ##   components for mixing weights, and for the parameters of the
  ##   mixture components.  Each paramter is stored in an array
  ##   matching the dimension of the parameter, plus 1.  Thus scalars
  ##   are stored in vectors.  Vectors are stored in matrices, and
  ##   matrices are stored in 3-way arrays.  The extra dimension
  ##   (always the first in each array) corresponds to MCMC iteration.

  if (inherits(mixture.component.specification, "MixtureComponent")) {
    ## Allow mixtures that use a single component.
    mixture.component.specification <- list(mixture.component.specification)
  }
  stopifnot(is.list(mixture.component.specification))
  stopifnot(all(sapply(mixture.component.specification,
                       inherits,
                       "MixtureComponent")))

  mixture.component.specification <-
    InferMixtureComponentNames(mixture.component.specification)

  if (is.null(weight.prior)) {
    if (is.null(state.space.size)) {
      stop("Error in FiniteMixture:  Either 'state.space.size'",
           "or 'weight.prior' must be specified.")
    }
    stopifnot(state.space.size > 0)
    weight.prior <- DirichletPrior(rep(1, state.space.size))
  }
  stopifnot(inherits(weight.prior, "DirichletPrior"))

  stopifnot(is.null(known.source) || is.numeric(known.source))
  if (is.numeric(known.source)) {
    if (min(known.source, na.rm = TRUE) == 1) {
      known.source <- known.source - 1
    }
    stopifnot(all(known.source < state.space.size, na.rm = TRUE))
    stopifnot(all(known.source >= 0, na.rm = TRUE))
  }

  ans <- .Call("boom_rinterface_fit_finite_mixture_",
               mixture.component.specification,
               weight.prior,
               niter,
               ping,
               known.source,
               seed,
               PACKAGE = "BoomMix")

  ans$mixture.component.specification <- mixture.component.specification

  class(ans) <- "FiniteMixture"
  return(ans)
}

InferMixtureComponentNames <- function(mixture.component.specification) {
  ## If the 'mixture.component.specification' list has names, and if
  ## some components lack the 'name' element, then use the names in
  ## 'mixture.component.specification' for the missing names.
  if (!is.null(names(mixture.component.specification))) {
    needs.name <- function(component) {
      return(is.null(component$name) || component$name == "")
    }
    for (i in 1:length(mixture.component.specification)) {
      if (needs.name(mixture.component.specification[[i]])) {
        mixture.component.specification[[i]]$name <-
          names(mixture.component.specification)[i]
      }
    }
  }
  return(mixture.component.specification)
}
