# Copyright 2018 Google LLC. All Rights Reserved.
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

StateSizes <- function(state.specification) {
  ## Returns a vector giving the number of dimensions used by each state
  ## component in the state vector.
  ## Args:
  ##   state.specification: a vector of state specification elements.
  ##     This most likely comes from the state.specification element
  ##     of a bsts.object
  ## Returns:
  ##   A numeric vector giving the dimension of each state component.
  state.component.names <- sapply(state.specification, function(x) x$name)
  state.sizes <- sapply(state.specification, function(x) x$size)
  if (any(is.na(state.sizes) ||
          any(is.null(state.sizes)) ||
          any(!is.numeric(state.sizes)))) {
    stop("One or more state components were missing the 'size' attribute.")
  }
  names(state.sizes) <- state.component.names
  return(state.sizes)
}
###----------------------------------------------------------------------
SuggestBurn <- function(proportion, bsts.object) {
  ## Suggests a size of a burn-in sample to be discarded from the MCMC
  ## run.
  ## Args:
  ##   proportion: A number between 0 and 1.  The fraction of the run
  ##     to discard.
  ##   bsts.object:  An object of class 'bsts'.
  ## Returns:
  ##   The number of iterations to discard.
  niter <- bsts.object$niter
  if (is.null(niter)) {
    ## Modern bsts objects will all have a 'niter' member.  Serialized
    ## bsts objects that were fit long ago expect niter to be the
    ## length of sigma.obs.
    niter <- length(bsts.object$sigma.obs)
  }
  if (is.null(bsts.object$log.likelihood)) {
    burn <- floor(proportion * niter)
  } else {
    burn <- SuggestBurnLogLikelihood(bsts.object$log.likelihood, proportion)
  }
  if (burn >= niter) {
    warning(paste0("SuggestBurn wants to discard everything\nn = ", niter,
                   "proportion = ", proportion, "."))
    if (niter > 0) {
      ## You have to keep at least one observation.
      burn <- niter - 1
    } else {
      ## You have to burn at least one observation.
      burn <- 1
    }
  }
  if (burn == 0) {
    ## This is to keep Kay's tests happy.
    burn <- 1
  }
  return(burn)
}
###----------------------------------------------------------------------
Shorten <- function(words) {
  ## Removes a prefix and suffix common to all elements of 'words'.
  ## Args:
  ##   words:  A character vector.
  ##
  ## Returns:
  ##   'words' with common prefixes and suffixes removed.
  ##
  ## Details:
  ##   The typical use case for this function is factor level names
  ##   that are (e.g.) names files from the same directory with
  ##   similar suffixes.  The common prefix (file path) gets removed,
  ##   as does the common suffix (.tex).
  ##
  ##   Shorten(c("/usr/common/foo.tex", "/usr/common/barbarian.tex")
  ##   produces c("foo", "barbarian")
  if (is.null(words)) return (NULL)
  stopifnot(is.character(words))
  if (length(unique(words)) == 1) {
    ## If all the words are the same don't do any shortening.
    return(words)
  }

  first.letters <- substring(words, 1, 1)
  while (all(first.letters == first.letters[1])) {
    words <- substring(words, 2)
    first.letters <- substring(words, 1, 1)
  }

  word.length <- nchar(words)
  last.letters <- substring(words, word.length, word.length)
  while (all(last.letters == last.letters[1])) {
    words <- substring(words, 1, word.length - 1)
    word.length <- word.length - 1
    last.letters <- substring(words, word.length, word.length)
  }

  return(words)
}

.SetTimeZero <- function(time0, y) {
  ## Args:
  ##   time0:  A timestamp to use as the answer, or NULL.
  ##   y: The time series being modeled.  If time0 is NULL and y is of type zoo
  ##     then the index of y[1] will be used as time0.
  ##
  ## Returns:
  ##   A timestamp of class POSIXt.
  if (is.null(time0)) {
    if (is.null(y)) {
      stop("You must supply time0 if y is missing.")
    }
    if (!inherits(y, "zoo")) {
      ## Note:  an xts object inherits from zoo.
      stop("You must supply 'time0' if y is not a zoo or xts object.")
    }
    times <- index(as.xts(y))
    tryCatch(time0 <- as.POSIXct(times)[1],
             error = simpleError(
               "The index of y could not be converted to POSIXt."))
  }
  stopifnot(inherits(time0, "POSIXt"))
  return(as.POSIXlt(time0))
}
