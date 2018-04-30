\alias{PlotMixtureParams}
\name{plot.mixture.params}
\title{Plot parameters of mixture components}
\Rdversion{1.0}

\description{Plots the parameters belonging to different mixture components.}

\usage{
PlotMixtureParams(model, stem, style = c("density", "ts", "box"),
                  colors = NULL, burn = 0, ...)

}

\arguments{

\item{model}{An object of class \code{FiniteMixture}}

\item{stem}{A character string giving the name of a parameter or mixture
 component to plot.  The stem can also either include the name of a
 particular parameter in a mixture component, or it can be the mixture
 component name, in which case all the parameters associated with that
 component will be plotted.}

\item{style}{A character string indicating the style of plot desired.
 "density" will plot kernel density estimates.  "ts" will plot the time
 series of MCMC draws.  "box" will produce side-by-side boxplots.}

\item{colors}{An optional character vector specifying the colors to
 use for the different mixture components.
}

\item{burn}{If burn > 0 then the first 'burn' MCMC draws will be
 discarded.}

\item{\dots}{Extra arguments passed to lower level plotting functions.}

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

## plot the mean parameters for the normal components.
PlotMixtureParams(model, "mu")

## Plot the rate parameters for the Poisson components.  The plot method
## for FiniteMixture calls PlotMixtureParams if the second argument is
## unrecognized.
plot(model, "lam", style = "ts")

## Plot the standard deviation parameters for the normal components.
plot(model, "sig", style = "box")


}
\keyword{models}
