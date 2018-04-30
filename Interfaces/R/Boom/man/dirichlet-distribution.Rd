\name{dirichlet-distribution}
\title{The Dirichlet Distribution}
\alias{ddirichlet}
\alias{rdirichlet}

\description{Density and random generation for the Dirichlet distribution.}

\usage{
ddirichlet(probabilities, nu, logscale = FALSE)
rdirichlet(n, nu)
}

\arguments{

  \item{probabilities}{A vector representing a discrete probability
    distribution, or a matrix where each row is a discrete probability
    distribution.  Zero probabilities are not allowed.}

  \item{nu}{The parameters of the Dirichlet distribution.  This can be a
    vector of positive numbers, interpretable as prior counts, of length
    matching the dimension of probabilities.  If \code{probabilities} is
    a matrix (or if \code{n > 1}) then \code{nu} can also be a matrix of
    the same dimension, in which case each row of \code{nu} is used to
    evaluate the corresponding row of \code{probabilities}.}

  \item{logscale}{Logical.  If TRUE then return the density on the log
    scale.  Otherwise return the density on the raw scale.}

  \item{n}{The number of desired draws.}
}

\value{\code{ddirichlet} returns a vector of density values, with one
  entry per row in \code{probabilities}.  \code{rdirichlet} returns a
  matrix (if \code{n > 1}) or a vector (if \code{n==1}) containing the
  draws from the Dirichlet distribution with the specified parameters.}

\details{

  The Dirichlet distribution is a generalization of the beta
  distribution.  Whereas beta distribution is a model for probabilities,
  the Dirichlet distribution is a model for discrete distributions with
  several possible outcome values.

  Let \eqn{\pi}{pi} denote a discrete probability distribution (a vector
  of positive numbers summing to 1), and let \eqn{\nu}{nu} be a vector
  of positive numbers (the parameters of the Dirichlet distribution),
  which can be thought of as prior counts.  Then the density of the
  Dirichlet distribution can be written

  \deqn{f(\pi) = \frac{\Gamma(\sum_i\nu_i)}{\prod_i \Gamma(\nu_i)}
    \prod_i \pi_i^{\nu_i-1}.}{%
    (gamma(sum(nu)) / prod(gamma(nu))) prod(pi^(nu - 1)).  }

}

\references{
  Gelman, Carlin, Stern, Rubin (2003), "Bayesian Data Analysis", Chapman
  and Hall.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}
