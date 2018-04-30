


\name{multinomial.mixture.component}
\alias{MultinomialMixtureComponent}
\title{Multinomial component for finite mixtures}
\Rdversion{1.0}
\description{For modeling data as a mixture of multinomial
  distributions.}

\usage{
MultinomialMixtureComponent(data, prior = NULL, group.id = NULL,
                            name = "",  ...)
}

\arguments{

  \item{data}{A \code{\link{factor}} specifying the data to be modeled.}

  \item{prior}{A prior distribution of class
    \code{\link[Boom]{DirichletPrior}} for the vector of
    multinomial probabilities.}

  \item{group.id}{A factor indicating the group to which each
    observation in \code{data} belongs.  This can be safely omitted when
    fitting finite mixtures (which assume observations are independent),
    but it is necessary when fitting hidden Markov and other model that
    consider within-subject variation.}

  \item{name}{The name of the mixture component.  This is optional, but
    recommended if more than one mixture component is used in a composite
    mixture model.}

  \item{...}{Extra arguments passed to the constructor of
    \code{\link[Boom]{DirichletPrior}}, if \code{prior} is omitted.}

}

\value{An object of class \code{MultinomialMixtureComponent}, which is a list
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
## fifty follow another.  We can't use this for clustering beacause a
## mixture of multinomials is just multinomial with different
## probabilities.
mix <- MultinomialMixtureComponent(x)
model <- FiniteMixture(mix, state.space.size = 2, niter = 1000)
par(mfrow = c(2,1))
plot.ts(model$prob.0, col = rainbow(3), plot.type = "single")  # Unhelpful!!
plot.ts(model$prob.1, col = rainbow(3), plot.type = "single")  # Unhelpful!!

## But multinomials can be useful in a hidden Markov model, or in
## conjunction with another model in a Finite mixture.
y1 <- rnorm(100, 0, 1)
y2 <- rnorm(50, 3, 1)
norm.mix <- NormalMixtureComponent(c(y1, y2))
model <- FiniteMixture(list(normal = norm.mix, multi = mix), 2,
                            niter = 1000)
par(mfrow = c(2, 1))
plot.ts(model$multi.prob.0, col = rainbow(3), plot.type = "single",
        main = "State 0")
plot.ts(model$multi.prob.1, col = rainbow(3), plot.type = "single",
        main = "State 1")
}
