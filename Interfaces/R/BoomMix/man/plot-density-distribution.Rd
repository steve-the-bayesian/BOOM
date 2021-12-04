\name{plot-density-distribution}
\alias{PlotDensityDistribution}
\title{Plot Density Distribution}
\Rdversion{1.0}

\description{For finite mixtures of univariate normals, plot the
  posterior distribution of the implied density curve using a dynamic
  distribution plot.
}

\usage{
PlotDensityDistribution(model,
                        burn = NULL,
                        xlim = NULL,
                        xlab = NULL,
                        ...)

}

\arguments{

\item{model}{An object of class \code{FiniteMixture}.  This function
  only applies to models with a single mixture component of class
  \code{\link{NormalMixtureComponent}}. }

\item{burn}{The number of MCMC iterations to be discarded as burn-in.
  If \code{NULL} then \code{\link[Boom]{SuggestBurnLogLikelihood}} will be
  called to suggest a burn-in value.}

\item{xlim}{Limits for the horizontal axis.}

\item{xlab}{Llabel for the horizontal axis.}

\item{\dots}{Extra arguments passed to \code{\link[Boom]{PlotDynamicDistribution}}.}

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
norm.mix <- NormalMixtureComponent(c(y1, y2))
model <- FiniteMixture(norm.mix, state.space.size = 2, niter = 100)

plot(model, "density")

}
\keyword{models}
