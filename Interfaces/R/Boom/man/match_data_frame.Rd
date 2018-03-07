\name{match_data_frame}
\title{MatchDataFrame}
\alias{MatchDataFrame}

\description{Given two data frames with the same data, but with rows and
  columns in potentially different orders, produce a pair of
  permutations such that data2[row.permutation, column.permutation]
  matches data1.}

\usage{
MatchDataFrame(data.to.match, data.to.permute)
}

\arguments{
  \item{data.to.match}{The data frame to be matched.}
  \item{data.to.permute}{The data frame to be permuted.}
}

\value{
  Returns a list with two elements.

  \item{column.permutation}{A vector of indices such that the columns of
    \code{data2[, column.permutation]} match the columns of
    \code{data1}.  The matching is based on column names.}

  \item{row.permutation}{A vector of indices such that the rows of
    \code{data2[row.permutation, column.permutation]} match the rows of
    \code{data1}.  The matching is done by converting rows to strings,
    and matching the strings.}
}

\author{
  Steven L. Scott \email{stevescott@google.com}
}

\examples{
 x1 <- data.frame(larry = rnorm(10), moe = 1:10, curly = rpois(10, 2))
 x2 <- x1[c(1:5, 10:6), c(3, 1, 2)]

 m <- MatchDataFrame(x1, x2)
 x2[m$row.permutation, m$column.permutation] == x1  ## all TRUE
}
