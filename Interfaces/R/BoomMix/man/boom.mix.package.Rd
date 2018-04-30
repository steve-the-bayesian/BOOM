


\alias{BoomMix-package}
\name{boom.mix.package}
\title{Finite mixture models}
\Rdversion{1.0}

\description{A package for fitting finite mixture models using Markov
  chain Monte Carlo.}

\section{Overview}{
  The BoomMix package provides support for Bayesian modeling with finite
  mixtures.  A finite mixture is a mixture of probability distributions
  of the form

  \deqn{ p(y) = w1 * p1(y) + w2 * p2(y) + ... + wk * pk(y) }

  where \eqn{w1 + ... + wk = 1}, \eqn{wi >= 0} and \eqn{p1, ..., pk} are
  probability distributions.  Finite mixtures are useful "nonparametric"
  models for data, as well as for unsupervised clustering of data.

}

\section{Mixture components for multivariate data}{

  There are several options if y is multivariate.  The package supplies
  a multivariate normal (mvn) mixture component, as well as options for
  regression and logistic regression.  It also allows for mixture
  composites, where if y = c(y1, y2), you can model eqn{pi(y) = pi1(y1)
  * pi2(y2)}.  For example, if y1 was continuous and y2 was count data
  you could set the first mixture component to Gaussian and the second
  to Poisson.

}

\section{The functions you need to know about}{

  The main function for fitting finite mixtures is called
  \code{\link{FiniteMixture}}.  You use \code{\link{FiniteMixture}} by
  providing a set of mixture components as the first argument, and a
  \code{\link[Boom]{DirichletPrior}} for the mixing weights as the second
  argument.

  The set of mixture components defines a mixture composite (as defined
  above), so if you just want a mixture of Gaussians your 'set' will
  just be a single \code{NormalMixtureComponent} or
  \code{MvnMixtureComponent}.

  There are several mixture components provided right now, and it is
  easy (for me) to add more (especially if they already exist on the
  BOOM side).

  \enumerate{
    \item{\code{\link{LogitMixtureComponent}}}
    \item{\code{\link{MultinomialMixtureComponent}}}
    \item{\code{\link{MvnMixtureComponent}}}
    \item{\code{\link{NormalMixtureComponent}}}
    \item{\code{\link{PoissonMixtureComponent}}}
    \item{\code{\link{RegressionMixtureComponent}}}
    \item{\code{\link{ZeroInflatedPoissonMixtureComponent}}}
    \item{\code{\link{ZeroInflatedLognormalMixtureComponent}}}
  }

}
