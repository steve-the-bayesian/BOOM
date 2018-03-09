% Copyright 2011 Google Inc. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{shorten}

\alias{Shorten}

\Rdversion{1.0}

\title{
  Shorten long names
}

\description{
  Removes common prefixes and suffixes from character vectors.
}

\usage{
  Shorten(words)
}

\arguments{
  \item{words}{A character vector to be shortened.}
}

\value{
  The argument \code{words} is returned, after common prefixes and
  suffixes have been removed.  If all arguments are identical then no
  shortening is done.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{bsts.mixed}}.
}

\examples{
  Shorten(c("/usr/common/foo.tex", "/usr/common/barbarian.tex"))
  # returns c("foo", "barbarian")

  Shorten(c("hello", "hellobye"))
  # returns c("", "bye")

  Shorten(c("hello", "hello"))
  # returns c("hello", "hello")

  Shorten(c("", "x", "xx"))
  # returns c("", "x", "xx")

  Shorten("abcde")
  # returns "abcde"
}

\keyword{character}
