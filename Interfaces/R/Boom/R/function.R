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
  dots <- list(...)
  if (length(dots) >= 1) {
    ## Handling extra arguments in C++ is hard, so wrap them up here in R.
    fwrapper <- function(x) {
      return(f(x, ...))
    }
    return(RVectorFunction(fwrapper))
  }
  ans <- list(
    function.name = deparse(substitute(f)),
    env = environment(f),
    thefun = f)
  class(ans) <- "RVectorFunction"
  return(ans);
}

