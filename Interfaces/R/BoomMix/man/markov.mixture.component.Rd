

\name{markov.mixture.component}
\alias{MarkovMixtureComponent}
\title{Markov component for finite mixtures}
\Rdversion{1.0}
\description{For modeling data as a mixture of Markov transitions.}

\usage{
MarkovMixtureComponent(data, prior = NULL, group.id = NULL, name = "",  ...)
}

\arguments{

  \item{data}{A \code{\link{factor}} specifying the data to be modeled.}

  \item{prior}{A prior distribution of class
    \code{\link[Boom]{MarkovPrior}} for the transition
    probability matrix, and the distribution of the initial state.}

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}

  \item{...}{Extra arguments passed to the constructor of
    \code{\link[Boom]{MarkovPrior}}, if \code{prior} is omitted.}

}

\value{An object of class \code{MarkovMixtureComponent}, which is a list
  containing information about the data and prior distribution.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{ \code{\link{FiniteMixture}}, \code{\link{BoomMix-package}}.  }

\examples{

x1 <- sample(c("a", "b", "c"), size = 100, replace = TRUE,
             prob = c(.2, .3, .5))
x2 <- sample(c("a", "b", "c"), size = 50, replace = TRUE,
             prob = c(.5, .3, .2))
x <- as.factor(c(x1, x2))

## The first 100 observations in x follow one pattern, while the next
## fifty follow another.
mix <- MarkovMixtureComponent(x)
model <- FiniteMixture(mix, state.space.size = 2, niter = 1000)
plot(model, "trans", style = "ts")
plot(model, "trans", style = "box")
plot(model, "trans", style = "den")

## Markov transitions can be useful in a hidden Markov model, or in
## conjunction with another model in a Finite mixture.
y1 <- rnorm(100, 0, 1)
y2 <- rnorm(50, 3, 1)
norm.mix <- NormalMixtureComponent(c(y1, y2))
model <- FiniteMixture(list(normal = norm.mix, markov = mix), 2,
                            niter = 1000)
plot(model, "normal")
plot(model, "trans")
}
