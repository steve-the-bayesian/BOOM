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


MarkovModulatedPoissonProcess <- function(point.process.list,
                                          process.specification,
                                          initial.state,
                                          mixture.components = NULL,
                                          known.source = NULL,
                                          niter,
                                          ping = niter / 10,
                                          seed = NULL) {
  ## Fits a Poisson cluster process using a posterior sampler run for
  ## 'niter' iterations.
  ## Args:
  ##   point.process.list: An object of class PointProcess, or a list
  ##     of such objects, giving the time points to be modeled.
  ##   process.specification: A list containing one or more
  ##     PoissonProcessComponent objects specifying the latent
  ##     processes, the processes spawned and killed by each type of
  ##     event, and the mixture components that model any associated
  ##     marks.
  ##   initial.state: A character vector containing a subset of the
  ##     names from process.specification defining any valid state in
  ##     the MMPP.  The model will use this information to determine
  ##     all the other states.
  ##   mixture.components: A list of mixture components created using
  ##     constructors from the BoomMix package.  These model the
  ##     'marks' associated with the point.process.list.  Multivariate
  ##     mark variables are modeled conditionally independently.  If
  ##     missing, no marks are assumed.
  ##   known.source: A list of character vectors with the same
  ##     dimensions as point.process.list.  Each vector contains zero
  ##     or more strings naming the processes or mixture components
  ##     that could have been the source for each event.  The elements
  ##     of a 'known.source' vector are names that can correspond to
  ##     either the names of individual point processes in
  ##     process.specification, or the names of mark models identified
  ##     in process.specification.
  ##       * If the names match one or more processess in
  ##         'process.specification' then those processes are assumed
  ##         to be the only candidates that could have produced the
  ##         corresponding event.
  ##       * If the names match one or more mark models from
  ##         'process.specification' then the processes associated
  ##         with those models are assumed to be the only candidates
  ##         that could have produced the corresponding event.
  ##       * If the vector of names is empty (character(0)) then it is
  ##         assumed that any of the processes in
  ##         process.specification could have produced the event, in
  ##         the sense that no processes are disallowed.
  ##       * If the entire known.source object is NULL then there are
  ##         no processes that should be disallowed as potential
  ##         sources for any of the points.
  ##     As a convenience, if the point.process.list contains a single
  ##     PointProcess, then known.source can be given as a character
  ##     vector.  In this case each entry of known.source identifies a
  ##     single PointProcess or mark model associated with a
  ##     particular time point.
  ##   niter:  The number of iterations to run the MCMC algorithm.
  ##   ping:  The desired frequency of update messages (in iterations).
  ##   seed: An optional integer used to seed the C++ random number
  ##     generator.  If missing the seed is set by the clock.
  ##
  ## Returns:
  ##   An object of class 'MarkovModulatedPoissonProcess', which is a
  ##   list containing the following elements
  ##   * foo
  ##   * bar
  ##   * baz
  CheckProcessSpecification(process.specification)
  stopifnot(is.character(initial.state))
  stopifnot(all(initial.state %in% names(process.specification)))
  stopifnot(is.numeric(niter))
  stopifnot(niter > 0)
  stopifnot(is.numeric(ping))

  if (inherits(point.process.list, "PointProcess")) {
    ## Support single subject point processes.
    point.process.list <- list(point.process.list)
  }
  stopifnot(is.list(point.process.list))
  stopifnot(all(sapply(point.process.list, inherits, "PointProcess")))

  if (inherits(mixture.components, "MixtureComponent")) {
    ## Allow mixtures that use a single component.
    mixture.components <- list(mixture.components)
  }
  if (!is.null(mixture.components)) {
    stopifnot(is.list(mixture.components))
    stopifnot(all(sapply(mixture.components, inherits, "MixtureComponent")))
    mixture.components <- InferMixtureComponentNames(mixture.components)
  }

  if (!is.null(known.source)) {
    if (length(point.process.list) == 1
        && is.character(known.source)
        && length(known.source) == length(point.process.list[[1]])) {
      known.source <- list(split(known.source,
                                 as.factor(1:length(known.source))))
    }
    stopifnot(is.list(known.source))
    stopifnot(length(known.source) == length(point.process.list))
    stopifnot(all(sapply(point.process.list, length) ==
                  sapply(known.source, length)))
  }

  model <- .Call("markov_modulated_poisson_process_wrapper_",
                 point.process.list,
                 process.specification,
                 initial.state,
                 mixture.components,
                 known.source,
                 niter,
                 ping,
                 seed)

  for (i in 1:length(model$prob.responsible)) {
    rownames(model$prob.responsible[[i]]) <- names(process.specification)
    rownames(model$prob.active[[i]]) <- names(process.specification)
  }
  if (!is.null(names(point.process.list))) {
    names(model$prob.active) <- names(point.process.list)
    names(model$prob.responsible) <- names(point.process.list)
  }
  if (length(model$prob.active) == 1) {
    model$prob.active <- model$prob.active[[1]]
    model$prob.responsible <- model$prob.responsible[[1]]
  }

  model$point.process.list <- point.process.list
  model$process.specification <- process.specification
  class(model) <- "MarkovModulatedPoissonProcess"
  return(model)
}

PoissonProcessComponent <- function(process,
                                    spawns = character(0),
                                    kills = character(0),
                                    mixture.component = character(0)) {
  ## Creates a Poisson process component that can be added to a
  ## MarkovModulatedPoissonProcess process specification.
  ##
  ## Args:
  ##   process: An object inheriting from the class
  ##     PoissonProcess, representing the PoissonProcess component to
  ##     be added.
  ##   spawns: A character vector naming the processes that are
  ##     activated by an event from 'process'.
  ##   kills: A character vector naming the processes that are
  ##     deactivated by an event from 'process'.
  ##   mixture.component: A string naming the mixture component that
  ##     models the marks for 'process'.  The definition of
  ##     the mixture component model family takes place in the call to
  ##     MarkovModulatedPoissonProcess.
  stopifnot(inherits(process, "PoissonProcess"))
  stopifnot(is.character(spawns))
  stopifnot(is.character(kills))
  stopifnot(is.character(mixture.component) || length(mixture.component) > 1)
  ans <- list(process = process,
              spawns = spawns,
              kills = kills,
              mixture.component = mixture.component)
  class(ans) <- "PoissonProcessComponent"
  return(ans)
}

CheckProcessSpecification <- function(process.specification) {
  ## Checks the valididity of process.specification for use in a
  ## MarkovModulatedPoissonProcess.

  stopifnot(is.list(process.specification))

  process.names <- names(process.specification)
  if (is.null(process.names)) {
    stop("The process.specification must be a list with named elements.")
  }

  stopifnot(all(nchar(process.names) > 0))

  ## Stop if any list elements do not inherit from
  ## PoissonProcessComponent.
  stopifnot(all(sapply(process.specification,
                       function(x) inherits(x, "PoissonProcessComponent"))))

  ## Get the unique list of process names, and the names of all
  ## processes killed and spawned.
  kills <- unique(unlist(lapply(process.specification,
                                function(x) x$kills)))
  spawns <- unique(unlist(lapply(process.specification,
                                 function(x) x$spawns)))

  ## Make sure any process named in 'kills' has an entry in
  ## 'process.specification'.
  bad.kills <- !(kills %in% process.names)
  if (any(bad.kills)) {
    stop("The following processes were named in 'kills', ",
         "but were not entered in process.specification:\n",
         paste(kills[bad.kills], collapse = "\n"),
         ".\n")
  }

  ## Make sure any process named in 'spawns' has an entry in
  ## 'process.specification'.
  bad.spawns <- !(spawns %in% process.names)
  if (any(bad.spawns)) {
    stop("The following processes were named in 'spawns', ",
         "but were not entered in process.specification:\n",
         paste(spawns[bad.spawns], collapse = "\n"),
         ".\n")
  }

  mixture.component.names <- lapply(process.specification,
                                    function(x) x$mixture.component)
  mix.name.size <- unique(sapply(mixture.component.names, length))
  if (any(mix.name.size > 1)) {
    stop("The following processes had multiple named mixture components:\n",
         paste(spawns[bad.spawns], collapse = "\n"),
         ".\n")
  }
  ok.mixture.names <- all(mix.name.size == 0) || all(mix.name.size == 1)
  if (!ok.mixture.names) {
    stop("Some components have a named mixture component, and some do not.")
  }
  return(invisible(NULL))
}

PlotProbabilityOfActivity <- function(mmpp,
                                      process,
                                      which.point.process = 1,
                                      from = NULL,
                                      to = NULL,
                                      xlab = "Time",
                                      ylab = "Probability of Activity",
                                      activity.color = "lightblue",
                                      ...) {
  ## Args:
  ##   mmpp:  The MarkovModulatedPoissonProcess model to be plotted.
  ##   process:  The name (or number) of the process to be plotted.
  ##   which.point.process: The index of the point process that was
  ##     used to fit the mmpp.  If the point.process.list had names,
  ##     then the name of the process can be used instead.

  pp <- mmpp$point.process.list[[which.point.process]]
  if (is.null(from)) {
    from <- pp$start
  }
  from <- as.POSIXct(from)
  if (is.null(to)) {
    to <- pp$end
  }
  to <- as.POSIXct(to)
  stopifnot(from <= to)

  event.times <- pp$events
  index <- (event.times <= to) & (event.times >= from)
  event.times <- event.times[index]
  times <- c(from, event.times, to)

  if (is.matrix(mmpp$prob.active)) {
    probs <- mmpp$prob.active[process, ]
  } else {
    probs <- mmpp$prob.active[[which.point.process]][process, ]
  }

  ## Each value of index corresponds to an event time.  The parallel
  ## value of probs is the probability that 'process' was active in
  ## the interval preceding that event time.  We also want to include
  ## the value of the next interval after the last true value in
  ## index, to capture the state between the last event and the end of
  ## the observation interval.
  numerical.index <- seq(along = index)[index]
  last.true.value <- max(numerical.index)
  probs <- probs[c(numerical.index, last.true.value + 1)]

  ## Set up the plotting region.
  plot(times,
       rep(0, length(times)),
       ylim = c(0, 1),
       type = "n",
       xlab = xlab,
       ylab = ylab,
       ...)
  x <- c(rep(times, each = 2), times[1])
  y <- c(0, rep(probs, each = 2), 0, 0)
  polygon(x, y, col = activity.color, border = NA)
  rug(event.times)
  abline(h=0, lty = 3)
}
