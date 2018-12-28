RVectorFunction <- function(f, ...) {
  ## Args:
  ##   f: A scalar-valued function of a vector-valued argument.  The function
  ##     can depend on other arguments as long as the vector valued argument
  ##     appears in the first position.
  ##   ...: Optional, named, extra arguments to be passed to f.  These arguments
  ##     are fixed at the time this object is created.  For the purpose of
  ##     evaluating f, these arguments do not update.
  ## 
  ## Returns:
  ##   A list containing the information needed to evaluate 'f' in C++ code.
  stopifnot(is.function(f))
  dots <- list(...)
  if (length(dots) >= 1) {
    ## Handling extra arguments in C++ is hard, so wrap them up here in R.
    fwrapper <- function(x) {
      return(f(x, ...))
    }
  } else {
    ## Wrapping the function like this also ensures we know its name.
    fwrapper <- f
  }
  ans <- list(
    function.name = ("fwrapper"),
    env = environment(fwrapper),
    thefun = fwrapper)
  class(ans) <- "RVectorFunction"
  return(ans);
}

