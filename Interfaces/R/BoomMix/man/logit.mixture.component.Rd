


\name{logit.mixture.component}
\alias{LogitMixtureComponent}
\alias{LogisticRegressionMixtureComponent}
\title{Logistic regression mixture component}
\Rdversion{1.0}

\description{For modeling data as a mixture of logistic regression models.}

\usage{

LogitMixtureComponent(formula, data, prior = NULL,
  contrasts = NULL, drop.unused.levels = TRUE, group.id = NULL,
  name = "", ...)

}

\arguments{

  \item{formula}{Model formula for the maximal model (with all variables
    included), parsed the same way as a call to \code{\link{glm}}.  The
    response variable can be a logical or numeric vector containing 0's
    and 1's, or it can be a two-column matrix where the first column
    contains counts of successes and the second contains counts of
    failures.}

  \item{data}{An optional \code{\link{data.frame}} containing variables
    used in \code{formula}.}

  \item{prior}{An object of class
    \code{\link[BoomSpikeSlab]{SpikeSlabPrior}} describing the logistic
    regression coefficients.}

  \item{contrasts}{ An optional list. See the \code{contrasts.arg} of
    \code{\link{model.matrix.default}}.  }

  \item{drop.unused.levels}{Logical indicating whether unobserved factor
    levels should be dropped from the model.}

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}

  \item{...}{Extra arguments passed to
    \code{\link[BoomSpikeSlab]{SpikeSlabPrior}}, if \code{prior} is
    omitted.}  }

\value{An object of class \code{LogitMixtureComponent}, which is a list
  containing information about the data and prior distribution.  }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{ \code{\link{FiniteMixture}}, \code{\link{BoomMix-package}}.  }
