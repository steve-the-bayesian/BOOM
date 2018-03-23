mscan <- function(fname, nc = 0, header = FALSE, burn = 0,
                  thin = 0, nlines = 0L, sep = "", ...) {
  ## Quickly scan a matrix of homogeneous data from a file.
  ##
  ## Args:
  ##   fname:  The name of the file from which to scan the data.
  ##   nc: the number of columns in the matrix to be read.  If zero then the
  ##     number of columns will be determined by the number of columns in the
  ##     first line of the file.
  ##   header: logical indicating whether the file contains a header row.
  ##   burn: An integer giving the number of initial lines of the matrix to
  ##     discard.
  ##   thin: An integer.  If thin > 1 then keep every thin'th line.  This is
  ##     useful for reading in very large files of MCMC output, for example.
  ##   nlines: If positive, the number of data lines to scan from the data file
  ##     (e.g. for an MCMC algorithm that is only partway done).  Otherwise the
  ##     entire file will be read.
  ##   sep:  Field separator in the data file.
  ##   ...:  Extra arguments passed to 'scan'.
  ##
  ## Returns:
  ##   A matrix containing values from the given data file.
  stopifnot(is.logical(header),
            length(header) == 1)
  stopifnot(is.character(fname),
            length(fname) == 1)
  stopifnot(is.numeric(nc),
            length(nc) == 1,
            nc >= 0)
  stopifnot(is.numeric(burn),
            length(burn) == 1,
            burn >= 0)
  stopifnot(is.numeric(thin),
            length(thin) == 1,
            thin >= 0)
  skip = as.numeric(header)
  if (header) {
    column.names <- scan(fname, nlines = 1, sep = sep, what = character(), ...)
    if (nc == 0) {
      nc <- length(column.names)
    }
    if (length(column.names) != nc) {
      stop("Specified number of columns: ",
           nc,
           " does not match the number of column names in the file header: ",
           length(column.names),
           ".")
    }
  } else {
    column.names <- NULL
  }

  if (nc == 0) {
    single.line <- scan(fname, skip = skip, nlines = 1, sep = sep, ...)
    nc <- length(single.line)
  }
  ans <- matrix(scan(fname, skip = skip, sep = sep, nlines = nlines, ...),
                ncol = nc,
                byrow = TRUE)
  if (burn > 0) ans <- ans[-(1:burn), , drop = FALSE]
  if (thin > 1) ans <- ThinMatrix(ans, thin)
  colnames(ans) <- column.names
  return(ans)
}
