slice.sampler <- function(logf, initial.guess, niter, ping = niter / 10, seed = NULL) {
  ## Args:
  ##   logf: The log density function you wish to sample using the slice
  ##     sampler.  This function should take a single argument (the location
  ##     where the density is to be evaluated) and return a scalar (the log
  ##     density at that location).
  ##   initial.guess:  The initial value for the slice sampling algorithm.
  ##   niter:  The desired number of MCMC iterations.
  ##   ping:  The frequency with which to print update messages.
  ##   seed: The random seed to use for the C++ random number generator, or
  ##     NULL.
  ##
  ## Returns:
  ##   A matrix of draws, with 'niter' rows, each of which is an MCMC draw.
  Boom::check.scalar.integer(niter)
  Boom::check.scalar.integer(ping)
  initial.logf <- logf(initial.guess)
  if (!is.finite(initial.logf)) {
    stop("The initial value passed to the slice sampler must yield a ",
      "finite density value.")
  }
  if (!is.null(seed)) {
    Boom::check.scalar.integer(seed)
  }

  rfun <- RVectorFunction(logf)
  rfun$function.name <- deparse(substitute(logf))
  
  ans <- .Call("boom_misc_slice_sampler_wrapper",
    rfun,
    as.numeric(initial.guess),
    as.integer(niter),
    as.integer(ping),
    as.integer(seed))
  if (exists("RVectorFunction_arg_", where = environment(logf))) {
    rm("RVectorFunction_arg_", envir = environment(logf))
  }
  return(ans)
}
