TestToString <- function() {
  m <- matrix(1:6, ncol = 2)
  printed.matrix <- ToString(m)
  checkEquals(printed.matrix,
       "     [,1] [,2]\n[1,]    1    4\n[2,]    2    5\n[3,]    3    6 \n")

  y <- c(1, 2, 3, 3, 3, 3, 3, 3)
  tab <- table(y)
  checkEquals(ToString(tab), "\nvalues:  1 2 3 \ncounts:  1 1 6  \n")
}
