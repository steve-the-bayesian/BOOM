ThinMatrix <- function(mat, thin) {
  ## Returns every thin'th row of the matrix mat.  This is useful for reducing
  ## MCMC output to reasonable size.
  ##
  ## Args:
  ##   mat:  The matrix to be thinned.
  ##   thin: The distance between kept lines from mat.  The larger the number
  ##     the fewer lines are kept.  E.g. ThinMatrix(mat, 10) keeps every 10th
  ##     line.  If thin <= 1 then no thinning is done.
  ##
  ## Returns:
  ##   The matrix mat, after discarding all but every 'thin' lines.
  stopifnot(is.matrix(mat))
  stopifnot(is.numeric(thin),
            length(thin) == 1)
  if (thin <= 1) return(mat)
  if (nrow(mat) < thin) return(head(mat, 1))
  top <- floor(nrow(mat) / thin)
  indx <- (1:top) * thin
  return(mat[indx, , drop = FALSE])
}
