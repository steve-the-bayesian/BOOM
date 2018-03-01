## A collection of utilities for formatting error messages.

ToString <- function(object, ...) {
  ## Args:
  ##   object:  An R object to be printed.
  ##   ...: extra arguments passed to 'print'
  ## Returns:
  ##   A character string, suitable for passing to an error message, containing
  ##   the value of the object, as rendered by the object's 'print' method.
  UseMethod("ToString")
}

ToString.default <- function(object, ...) {
  ## Args:
  ##   object:  An R object to be printed.
  ##   ...: extra arguments passed to 'print'
  ## Returns:
  ##   A character string, suitable for passing to an error message, containing
  ##   the value of the object, as rendered by the object's 'print' method.
  output <- capture.output(print(object, ...))
  return(paste(paste(output, collapse = "\n"), "\n"))
}

ToString.table <- function(object, ...) {
  ## Args:
  ##   object:  A table whose entries are to be written to a string.
  ##   ...: extra arguments passed to 'print'
  ## Returns:
  ##   A character string containing the formatted entries of 'object', suitable
  ##   for passing to an error message.
  stopifnot(is.table(object))
  tab.string <- capture.output(print(object, ...))[-1]
  tab.string[1] <- paste("\nvalues: ", tab.string[1])
  tab.string[2] <- paste("counts: ", tab.string[2], "\n")
  return(paste(tab.string, collapse = "\n"))
}
