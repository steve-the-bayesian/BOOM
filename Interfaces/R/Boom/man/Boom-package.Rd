\name{Boom-package}
\alias{Boom-package}
\alias{Boom}
\docType{package}
\title{
  Boom
}

\description{ The Boom package provides access to the C++ BOOM library
  for Bayesian computation.  }

\details{
  \subsection{Installation note for Linux users}{
    If you are installing Boom using \code{\link{install.packages}} on a
    Linux machine (and thus compiling yourself) you will almost certainly
    want to set the \code{Ncpus} argument to a large number.  Windows and
    Mac users can ignore this advice.
  }

  The main purpose of the Boom package is not to be used directly, but
  to provide the BOOM C++ library for other packages to link against.
  The Boom package provides additional utility code for C++ authors to
  use when writing R packages with C++ internals.  These are described
  in \code{.../inst/include/r_interface/boom_r_tools.hpp} among the
  package's include files.

  Boom provides a collection of R functions and objects to help users
  format data in the manner expected by the underlying C++ code.
  Standard distributions that are commonly used as Bayesian priors can
  be specified using \code{\link{BetaPrior}}, \code{\link{GammaPrior}},
  etc.

  Boom provides a set of utilities helpful when writing unit tests for
  Bayesian models.  See \code{\link{CheckMcmcMatrix}} and
  \code{\link{CheckMcmcVector}} for MCMC output, and functions like
  \code{\link{check.probability.distribution}} for checking function
  inputs

  Boom provides a collection of useful plots (using base R graphics)
  that have proven useful for summarizing MCMC output.  See
  \code{\link{PlotDynamicDistribution}}, \code{\link{PlotManyTs}},
  \code{\link{BoxplotTrue}}, and other code in the index with
  \code{Plot} in the title.
}

\keyword{ package }

\seealso{
  Please see the following pacakges
  \itemize{
  \item \code{bsts}
  \item \code{CausalImpact}
  }
}
