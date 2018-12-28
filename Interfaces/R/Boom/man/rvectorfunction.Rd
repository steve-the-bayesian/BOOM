% Copyright 2018 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{rvectorfunction}
\title{RVectorFunction}
\alias{RVectorFunction}

\description{A wrapper for passing R functions to C++ code.}

\usage{
  RVectorFunction(f, ...)
}

\arguments{

  \item{f}{A scalar-valued function of a vector-valued argument.  The
    function can depend on other arguments as long as the vector valued
    argument appears in the first position.  }

  \item{\dots}{Optional, named, extra arguments to be passed to f.
    These arguments are fixed at the time this object is created.
    For the purpose of evaluating f, these arguments do not update.}
}

\value{
  A list containing the information needed to evaluate the function f in
  C++ code.
}

\details{ The Boom library can handle the output of this function as a
  C++ function object.  Note that evaluating R functions in C is no
  faster than evaluating them in R, but a wrapper like this is useful
  for interacting with C and C++ libraries that expect to operate on C
  and C++ functions.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
