\name{unit.testing}
\Rdversion{1.0}
\alias{PlotUnitTestOutput}
\alias{ReadUnitTestOutput}

\title{
  Utilities for Unit Test Output
}
\description{
  BOOM's C++ unit tests often write MCMC output to files.  The Boom R
  package includes these tools for inspecting the output.
}
\usage{
PlotUnitTestOutput(fname,
                   dim = NULL,
                   truth = TRUE,
                   thin = 0,
                   burn = 0,
                   style = c("ts", "box"),
                   header = FALSE,
                   ...)

ReadUnitTestOutput(fname, dim=NULL, thin=0, burn=0, ...)
}

\arguments{

\item{fname}{The name of a numeric text file containing MCMC output.
  Each row in the file is one MCMC draw of a parameter.}

\item{dim}{Either \code{NULL} or a vector of dimensions describing the
  shape of one row in \code{fname}.}

\item{thin}{If greater than zero, read every \code{thin}'th row of the
  file.  I.e. if \code{thin = 10} then read every 10th row.}

\item{burn}{The number of initial rows from the file to remove as MCMC
  burn-in.}

\item{truth}{If \code{TRUE} then assume the first line of \code{fname}
  contains true values of the simulated parameters.  If present these
  will be highlighted in the plots.}

\item{style}{Indicates whether time series plots or boxplots should be
  used to show the output.}

\item{header}{If TRUE then assume \code{fname} contains variable names
  in the first row.}

\item{\dots}{Extra arguments passed to \code{mscan} for reading the and 'axis'.}

}

\author{
  Steven L. Scott
}
\seealso{
  \code{\link{PlotManyTs}}, \code{\link{BoxplotTrue}}, \code{\link{BoxplotMcmcMatrix}}
}


\keyword{ hplot }
