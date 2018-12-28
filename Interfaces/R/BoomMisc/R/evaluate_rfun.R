.EvaluateFunction <- function(f, x, ...) {
  ## A utility function for testing RVectorFunction.  This is intended to be
  ## useful outside a testing context.
  function.wrapper <- RVectorFunction(f, ...)
  return(.Call("boom_evaluate_r_function", function.wrapper, x))
}
