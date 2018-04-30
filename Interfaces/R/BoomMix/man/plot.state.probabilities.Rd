\alias{PlotStateProbabilities}
\name{plot.state.probabilities}
\title{Plot class membership probabilities}
\Rdversion{1.0}

\description{Plot the posterior probability that each observation
  belongs to each mixture component.}

\usage{
PlotStateProbabilities(model,
                       group.id = NULL,
                       colors = NULL,
                       xlab = "Observation",
                       ylab = "Probability",
                       components = NULL,
                       ...)

}

\arguments{

\item{model}{An object of class \code{FiniteMixture}}

\item{group.id}{The specific group for which state probabilities should
  be plotted.  If \code{NULL} then all groups will be plotted.  If you
  did not specify \code{group.id} when fitting \code{model} then this
  parameter can be ignored.}

\item{colors}{A vector of colors to use for the component plots.  This
  can help make associations between these plots and (e.g.) those
  produced by \code{\link{PlotMixtureParams}}.}

\item{xlab}{  The label for the horizontal axis.}

\item{ylab}{The label for the vertical axis.}

\item{components}{A numeric vector listing which components should be
  plotted.  Components are numbered starting from 0.}

\item{\dots}{Extra arguments passed to \code{\link{plot}}.}

}

\details{ Creates a plot (in the requested style) on the current
graphics device. }

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomMix-package}}.
}

\examples{
y1 <- rnorm(100, 0, 1)
y2 <- rnorm(50, 3, 1)
z1 <- rpois(100, 3)
z2 <- rpois(50, 6)
norm.mix <- NormalMixtureComponent(c(y1, y2))
pois.mix <- PoissonMixtureComponent(c(z1, z2))
model <- FiniteMixture(list(norm.mix, pois.mix),
                       weight.prior = DirichletPrior(rep(1, 2)),
                       niter = 100,
                       ping = 10)

PlotStateProbabilities(model)
plot(model, "state.probabilities")

}
\keyword{models}
