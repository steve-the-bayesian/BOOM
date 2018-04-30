\name{traceproduct}
\title{Trace of the Product of Two Matrices}
\alias{TraceProduct}

\description{Returns the trace of the product of two matrices.}

\usage{
TraceProduct(A, B, b.is.symmetric = FALSE)
}

\arguments{
  \item{A}{The first matrix in the product.}
  \item{B}{The second matrix in the product.}
  \item{b.is.symmetric}{Logical.  A \code{TRUE} value indicates that
    \code{B} is a symmetric matrix.  A slight computational savings is
    possible if \code{B} is symmetric.}
}

\value{
  Returns a number equivalent to \code{sum(diag(A \%*\% B))}.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
