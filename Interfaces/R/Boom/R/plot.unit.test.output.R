ReadUnitTestOutput <- function(fname, dim=NULL, thin=0, burn=0, ...){

  ## Args:
  ##   fname:  a numeric text file where each line is a vectorized array.
  ##   dim: An integer vector giving the dimension of the random variables
  ##     in a single row of the array.
  ##   thin:  If thin > 1 then read every 'thin'th' line of the file.
  ##     For example if thin = 10 every 10th line will be read.
  ##   burn: The number of initial lines in 'fname' to discard as burn-in.
  ##   ...:  Extra arguments passed to 'mscan'.

  ## Returns:
  ##   an array of dimension c(nr, dim[1], dim[2], ...,
  ##   dim[p]) where nr is the number of rows in the input file.

  tmp <- mscan(fname, thin=thin, burn=burn, ...)
  if (is.null(dim)) {
    return(tmp)
  }
  nr <- nrow(tmp)
  p <- length(dim)
  x <- array(t(tmp), dim= c(rev(dim), nr))
  aperm(x, (p+1):1)
}

PlotUnitTestOutput <- function(fname,
                               dim = NULL,
                               truth = TRUE,
                               thin = 0,
                               burn = 0,
                               style = c("ts", "box"),
                               header = FALSE,
                               ...) {
  ## Plot the results of an MCMC run stored as text data in a file.
  ##
  ## Args:
  ##   fname: The name of the file containing the output.  Each row is one Monte
  ##     Carlo draw.
  ##   dim: If non-NULL, a vector of dimensions describing the shape of one row
  ##     in the output file.  For example, if the output contains a vectorized
  ##     5x5 varaince matrix then dim=c(5, 5).
  ##   truth: If TRUE then the first line of 'fname' contains true values
  ##     of the simulated parameters.  If present these will be highlighted in
  ##     the plots.
  ##   thin:  If > 0 then only read in every "thin'th" line of 'fname'.
  ##     For example if thin = 10 every 10th line will be read.
  ##   burn:  If > 0 then
  ##   style:  One of "ts" (for time series plots) or "box" (for box plots).
  ##   ...: Extra arguments passed to plotting functions.

  draws <- ReadUnitTestOutput(fname, dim = dim, thin = thin, burn = burn, header=header)

  if (truth > 0) {
    if (is.null(dim) || length(dim) == 1) {
      truth <- draws[1, ]
      draws <- draws[-1, ]
    } else if (length(dim) == 2) {
      truth <- draws[1, , ]
      draws <- draws[-1, , ]
    } else if (length(dim) == 3) {
      truth <- draws[1, , , ]
      draws <- draws[-1, , , ]
    } else {
      stop("Arrays of 4 or more dimensions are not supported.")
    }
  }

  style <- match.arg(style)
  if (style == "ts") {
    PlotManyTs(draws, truth=truth, refline = 0, ...)
  } else if (style == "box") {
    if (is.null(dim) || length(dim) == 1) {
      BoxplotTrue(draws, truth=truth)
    } else if (length(dim) == 2) {
      BoxplotMcmcMatrix(draws, truth=truth, ...)
    }
  }
}
