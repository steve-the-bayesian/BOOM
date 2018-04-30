\alias{InferMixtureComponentNames}
\name{infer.mixture.component.names}
\title{Infer mixture component names}
\Rdversion{1.0}

\description{ Infer the name of a mixture compnent from its name in a
  list.  This function is intended for library authors.  It is unlikely
  to be called directly by the user.  }

\usage{
InferMixtureComponentNames(mixture.component.specification)
}

\arguments{

  \item{mixture.component.specification}{Either a single object, or a
    list of objects, inheriting from class \code{MixtureComponent}.  See
    \code{\link{BoomMix-package}} for comments on how to structure
    mixture components.}

}

\value{ Each element of \code{mixture.component.specification} is
  examined.  If the component lacks a \code{name} element then an
  attempt is made to infer the component's name from the names attribute
  of the \code{mixture.component.specification} list.  If successful a \code{name}
  attributed is added.  The modified version of
  \code{mixture.component.specification} is returned}


\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomMix-package}}.
}

\examples{
y <- rnorm(100, 0, 1)
mix <- NormalMixtureComponent(y)
components <- list(normal = mix)
components[[1]]$name  ## NULL
components <- InferMixtureComponentNames(components)
components[[1]]$name ## "normal"
}

\keyword{models}
