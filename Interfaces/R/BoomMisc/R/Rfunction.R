RVectorFunction <- function(f) {
  ## Args:
  ##   f:  A scalar-valued function of a single vector-valued argument.
  ##
  ## Returns:
  ##   A list containing the information needed to evaluate 'f' in C++ code.
  ans <- list(
    function.name = deparse(substitute(f)),
    env = environment(f))
  class(ans) <- "RVectorFunction"
  return(ans);
}
