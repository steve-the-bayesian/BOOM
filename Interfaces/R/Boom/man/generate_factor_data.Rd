\name{GenerateFactorData}
\title{Generate a data frame of all factor data}
\alias{GenerateFactorData}

\description{ This function is mainly intended for example code and unit
  testing.  It generates a \code{data.frame} containing all factor data.}

\usage{
  GenerateFactorData(factor.levels.list, sample.size)
}

\arguments{

  \item{factor.levels.list}{A list of character vectors giving factor
    level names.  The names attribute of this list becomes the set of
    variables names for the return data frame.}

  \item{sample.size}{The desired number of rows in the returned data frame.}
}

\examples{
  foo <- GenerateFactorData(list(a = c("foo", "bar", "baz"),
                                 b = c("larry", "moe", "curly", "shemp")),
                            50)

  head(foo)
#     a     b
# 1 bar curly
# 2 foo curly
# 3 bar   moe
# 4 bar   moe
# 5 baz curly
# 6 bar curly

}

\author{
  Steven L. Scott \email{stevescott@google.com}
}
