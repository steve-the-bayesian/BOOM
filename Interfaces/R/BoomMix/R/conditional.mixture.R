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

ConditionalMixture <- function(mixing.distribution.formula,
                               mixture.component.specification,
                               state.space.size = NULL,
                               mixing.distribution.prior = NULL,
                               niter,
                               ping = niter / 10,
                               known.source = NULL,
                               data,
                               contrasts = NULL,
                               expected.model.size = 1.0 * state.space.size,
                               seed = NULL) {
  ## Fits a finite mixture model of the form y ~ w_1(x) f_1(y) + ... +
  ## w_k(x)f_k(y).  The f's all share a common model family determined
  ## by mixture.component.specification, and the w(x)'s are determined
  ## by a multinomial logistic regression.
  ## Args:
  ##   mixing.distribution.formula: A formula for determining the
  ##     design matrix for the conditional mixing weights.  The
  ##     formula should not contain a response variable.
  ##   mixture.component.specification: A list of objects inheriting
  ##     from class MixtureComponent.  If the list would otherwise be
  ##     of length 1 then this can be a single object of class
  ##     MixtureComponent.  MixtureComponent objects contain their
  ##     data and a prior specification.
  ##   state.space.size: The number of states in the mixture chain.
  ##     In the typical way of invoking ConditionalMixture, the user
  ##     will specify 'state.space.size' and specify
  ##     'mixing.distribution.prior' mainly through the ... argument.
  ##     If 'mixing.distribution.prior' is passed directly then
  ##     state.space.size is inferred from
  ##     'mixing.distribution.prior'.
  ##   mixing.distribution.prior: An object of class
  ##     IndependentSpikeSlabPrior (or SpikeSlabPrior) determining the
  ##     prior on the coefficients of the mixing weights.  The
  ##     dimension of the prior must be an integer multiple
  ##     (state.space.size - 1) of the dimension of the design matrix
  ##     for the mixture components.  Alternatively,
  ##     mixing.distribution.prior can be left NULL, and a default
  ##     prior will be constructed using 'expected.model.size' and
  ##     'state.space.size'.
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
  ##   data: An optional data frame containing the variables
  ##     referenced in 'formula'.
  ##   contrasts: an optional list. See the 'contrasts.arg' of
  ##     ‘model.matrix.default’.
  ##   expected.model.size: The expected number of nonzero
  ##     coefficients in the mixing distribution.  This is ignored if
  ##     mixing.distribution.prior is specified.
  ##   seed: Seed to use for the C++ random number generator.  NULL or
  ##     an int.  If NULL, then the seed will be taken from the global
  ##     .Random.seed object.
  ##
  ## Returns:
  ##   An object of class ConditionalMixture, which is a list with
  ##   components for the mixing distribution, and for the parameters
  ##   of the mixture components.  Each paramter is stored in an array
  ##   matching the dimension of the parameter, plus 1.  Thus scalars
  ##   are stored in vectors.  Vectors are stored in matrices, and
  ##   matrices are stored in 3-way arrays.  The extra dimension
  ##   (always the first in each array) corresponds to MCMC iteration.
  if (inherits(mixture.component.specification, "MixtureComponent")) {
    mixture.component.specification <- list(mixture.component.specification)
  }
  stopifnot(is.list(mixture.component.specification))
  stopifnot(all(sapply(mixture.component.specification,
                       inherits,
                       "MixtureComponent")))
  mixture.component.specification <-
    InferMixtureComponentNames(mixture.component.specification)

  mixture.formula <- match.call(expand.dots = FALSE)

  mixture.model.frame.prototype <-
    mixture.formula[c(1L,
                      match(c("mixing.distribution.formula",
                              "data",
                              "na.action"),
                            names(mixture.formula),
                            0L))]
  formula.position <- grep("mixing.distribution.formula",
                           names(mixture.model.frame.prototype))
  names(mixture.model.frame.prototype)[formula.position] <- "formula"

  mixture.model.frame.prototype[[1L]] <- quote(stats::model.frame)
  mixture.model.frame <- eval(mixture.model.frame.prototype, parent.frame())
  mixture.model.terms <- attr(mixture.model.frame, "terms")
  mixture.design.matrix <- model.matrix(mixture.model.terms,
                                        mixture.model.frame,
                                        contrasts)

  if (is.null(mixing.distribution.prior)) {
    if (is.null(state.space.size)) {
      stop("One of 'state.space.size' or ",
           "'mixing.distribution.prior' must be specified")
    }
    ## TODO(stevescott): replace this with a
    ## MultinomialLogitSpikeSlabPrior once that gets developed.
    mixing.distribution.prior <-
      IndependentSpikeSlabPrior(y <- rep(0, nrow(mixture.design.matrix)),
                                x = mixture.design.matrix,
                                sdy = 1,
                                expected.model.size = expected.model.size)
    mixing.distribution.prior$mu <-
      rep(mixing.distribution.prior$mu,
          (state.space.size - 1))
    mixing.distribution.prior$prior.inclusion.probabilities <-
      rep(mixing.distribution.prior$prior.inclusion.probabilities,
          (state.space.size - 1))
    mixing.distribution.prior$prior.variance.diagonal <-
      rep(mixing.distribution.prior$prior.variance.diagonal,
          (state.space.size - 1))
  }
  stopifnot(inherits(mixing.distribution.prior, "SpikeSlabPriorBase"))

  ans <- .Call("boom_rinterface_fit_conditional_mixture_",
               mixture.component.specification,
               mixing.distribution.prior,
               mixture.design.matrix,
               niter,
               ping,
               known.source,
               seed)
  class(ans) <- c("ConditionalMixture", "FiniteMixture")
  return(ans)
}
