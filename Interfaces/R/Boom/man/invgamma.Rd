\name{invgamma}
\title{Inverse Gamma Distribution}
\alias{rinvgamma}
\alias{pinvgamma}
\alias{qinvgamma}
\alias{dinvgamma}

\description{Density, distribution function, quantile function, and
  random draws from the inverse gamma distribution.}

\usage{
dinvgamma(x, shape, rate, logscale = FALSE)
pinvgamma(x, shape, rate, lower.tail = TRUE, logscale = FALSE)
qinvgamma(p, shape, rate, lower.tail = TRUE, logscale = FALSE)
rinvgamma(n, shape, rate)
}

\arguments{

  \item{x}{A vector of deviates where the density or distribution
    function is to be evaluated.}

  \item{p}{A vector of probabilities representing CDF values (if
    \code{lower.tail == TRUE}) or survivor function values (if
    \code{lower.tail == FALSE}) from the inverse gamma distribution.  }

  \item{n}{The desired number of draws from the inverse gamma distribution.}

  \item{shape}{The shape parameter.}

  \item{rate}{The 'rate' parameter.  NOTE: The term 'rate' is used to
    match the corresponding parameter in \code{\link{rgamma}}. Much
    of the rest of the world calls this the 'scale' parameter.}

  \item{logscale}{Logical.  If \code{TRUE} then probabilities or density
    values are interpreted on the log scale.  Otherwise the scale is the
    probability or probability density scale.}

  \item{lower.tail}{Logical.  If TRUE then cumulative probabilities are
    measured from zero, as in a CDF.  If FALSE then cumulative are
    measured from infinity, as in a survivor function.}
}

\value{
  \itemize{
  \item{\code{rinvgamma} returns draws from the distribution.}
  \item{\code{dinvgamma} returns the density function.}
  \item{\code{pinvgamma} returns the cumulative distribution function
    (or survivor function, if \code{lower.tail == FALSE}).}
  \item{\code{qinvgamma} returns quantiles from the distribution.
    \code{qinvgamma} and \code{pinvgamma} are inverse functions.}
}
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
