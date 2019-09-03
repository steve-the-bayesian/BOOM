\name{mscan}
\title{Scan a Matrix}
\alias{mscan}

\description{Quickly scan a matrix from a file.}

\usage{
  mscan(fname, nc = 0, header = FALSE, burn = 0, thin = 0, nlines = 0L,
        sep = "", \dots)
}

\arguments{
  \item{fname}{The name of the file from which to scan the data.}
  \item{nc}{The number of columns in the matrix to be read.  If zero then the
   number of columns will be determined by the number of columns in the
   first line of the file.}
  \item{header}{logical indicating whether the file contains a header row.}
  \item{burn}{An integer giving the number of initial lines of the matrix to discard.}
  \item{thin}{An integer.  If thin > 1 then keep every thin\'th line.  This is
    useful for reading in very large files of MCMC output, for example.}
  \item{nlines}{If positive, the number of data lines to scan from the data file
    (e.g. for an MCMC algorithm that is only partway done).  Otherwise the
    entire file will be read.}
  \item{sep}{ Field separator in the data file.}
  \item{\dots}{Extra arguments passed to 'scan'.}
}

\value{
  The matrix stored in the data file.
}

\details{
  This function is similar to \code{\link{read.table}}, but scanning a
  matrix of homogeneous data is much faster because there is much less
  format deduction.
}

\examples{

filename <- file.path(tempdir(), "example.data")
cat("foo bar baz", "1 2 3", "4 5 6", file = filename, sep = "\n")
m <- mscan(filename, header = TRUE)
m
##      foo bar baz
## [1,]   1   2   3
## [2,]   4   5   6
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
