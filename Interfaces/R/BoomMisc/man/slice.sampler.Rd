\name{slice.sampler}
\Rdversion{1.0}
\alias{slice.sampler}
\title{
  Slice Sampling
}
\description{
  A slice sampler that applies the univariate slice sampling algorithm
  to each coordinate of a log density function, as part of a scalar
  Gibbs sampling strategy.
}
\usage{
  slice.sampler(logf, initial.guess, niter, ping = niter / 10, seed = NULL)
}

\arguments{

  \item{logf}{ The log density function you wish to sample using the
    slice sampler.  This function should take a single argument (the
    location where the density is to be evaluated) and return a scalar
    (the log density at that location).  }
     
  \item{initial.guess}{
    The initial value for the slice sampling algorithm.
  }

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{The frequency with which to print status updates on the
  MCMC.  To skip update messages set \code{ping < 0}. }
  
  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}

}

\details{

  The supplied function is passed to C++ code implementing the slice
  sampler, which will sample one variable at a time (like early versions
  of BUGS).  This is not a particularly efficient approach,
  computationally speaking, but it should be good for small problems and
  early proofs of concept.

  If you can, specify logf using sufficient statistics.  That will speed
  things up a lot.

  All variables other than the function argument to the supplied
  function must be visible in the function's environment (most likely
  the global R environment).
}

\value{
  Returns a matrix, the rows of which are the MCMC draws.
}

\references{
  Neal (2003)
}

\author{
  Steven L. Scott
}


\examples{
n <- 30
y <- rgamma(n, 3, 7)

logf1 <- function(theta) {
  ## Log posterior of the gamma distribution, with standard normal priors on
  ## log(a/b) and log(a).  theta = log(a/b), log(a)
  a <- exp(theta[2])
  mu <- exp(theta[1])
  b <- a / mu
  ans <- sum(dgamma(y, a, b, log = TRUE))
  ans <- ans + dnorm(theta[1], log = TRUE) + dnorm(theta[2], log = TRUE)
  return(ans)
}

## The first log density evaluates the gamma likelihood using raw data.
draws <- slice.sampler(logf1, c(0, 0), 1000)

GammaSuf <- function(y) {
  return(list(n = length(y),
              sum.y = sum(y),
              sumlog.y = sum(log(y))))
}
suf <- GammaSuf(y)

## The second log density evaluates the gamma likelihood using
## sufficient statistics, which is MUCH FASTER when n gets big.

logf2 <- function(theta) {
  ## This is another version of logf that uses sufficient statistics.
  a <- exp(theta[2])
  mu <- exp(theta[1])
  b <- a / mu
  ans <- suf$n * a * log(b) - suf$n * lgamma(a) + (a-1) * suf$sumlog.y - b * suf$sum.y
  ans <- ans + dnorm(theta[1], log = TRUE) + dnorm(theta[2], log = TRUE)
}

Gammafy <- function(draws) {
  if (!is.matrix(draws)) {
    draws <- matrix(draws, nrow = 1)
  }
  a <- exp(draws[, 2])
  mu <- exp(draws[, 1])
  b <- a/mu
  return(cbind(a, b))
}

more.draws <- slice.sampler(logf2, c(0, 0), niter = 1000)

opar <- par(mfrow = c(1, 2))
plot(Gammafy(draws))
plot(Gammafy(more.draws))
par(opar)

## Now set n = 5000 any try again!!

}
\keyword{models}
\keyword{regression}
