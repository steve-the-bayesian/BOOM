\name{partial-dependence}
\alias{BartPartialDependencePlot}

\title{partial dependence plots}
\Rdversion{1.0}
\description{
  Partial dependence plots for Bart models.
}

\usage{
BartPartialDependencePlot(model,
                          which.variable,
                          burn = SuggestBartBurn(model),
                          data.fraction = .2,
                          thin = 10,
                          mean.only = FALSE,
                          show.points = FALSE,
                          xlab = NULL,
                          ylab = NULL,
                          ylim = NULL,
                          ...)
}

\arguments{
  \item{model}{The model object returned by BoomBart.}

  \item{which.variable}{Either the name of the variable for which a
    partial dependence plot is desired, or its index (column number) in
    the design matrix.}

  \item{burn}{The number of initial MCMC iterations to discard as
    burn-in.}

  \item{data.fraction}{ The fraction of observations in the design
    matrix to use when constructing the partial dependence plot.  A
    random sub-sample of this fraction will be taken (without
    replacement).}

  \item{thin}{The frequency of MCMC iterations to keep after burn-in.
    For example, setting \code{thin = 10} would keep every 10th
    iteration.}

  \item{mean.only}{Logical.  If \code{TRUE} then only the mean is
    plotted at each point.  If \code{FALSE} then the posterior of the
    function value is plotted.}

  \item{show.points}{If \code{TRUE} then the scatterplot of x vs y is
    added to the graph.  Otherwise the points are left off.  }

  \item{xlab}{Label for the X axis.  \code{NULL} produces a default label.  Use
    "" for no label.}

  \item{ylab}{Label for the Y axis.  \code{NULL} produces a default label.
    Use "" for no label.}

  \item{ylim}{Limits on the vertical axis.  If \code{NULL} then the plot
    will default to its natural vertical limits.}

  \item{...}{Extra arguments are passed either to \code{\link{plot}}
  (if \code{mean.only == TRUE}) or
  \code{\link[Boom]{PlotDynamicDistribution}} (otherwise).}

}

\value{Invisibly returns a list containing the posterior of the
     distribution function values at each 'x' value, and the x's where
     the distribution is calculated. }

\details{
  Draws a partial dependence plot on the current graphics device.  A
  partial dependence plot shows the relationship between X and Y,
  averaging over the distribution of other X's.  It is a generalization
  of an "added variable plot" or a "leverage plot."

  To do the averaging, this function will take the design matrix on
  which the model was fit (which is stored in the \code{model} object),
  and vary the \code{which.variable} variable from its smallest to
  largest value.  At each abscissa, a prediction is made for each row of
  the design matrix, with the current abscissa substituted for the
  element indicated by \code{which.variable}.  The predictions are then
  averaged to get the "Y" value for the plot at that point.  This
  procedure is done once per MCMC iteration, to produce a posterior
  distribution of partial dependence function values.

  Partial dependence plots can expensive to produce.  You can control
  the cost by thinning the MCMC output (using the \code{thin} parameter)
  and by taking a random subset of the data (using the
  \code{data.fraction} parameter).
}

\references{
Chipman, George, McCulloch (2010) BART: Bayesian additive regression
trees.  Annals of Applied Statistics.  Volume 4, Number 1. 266 -- 298.

Friedman, J.H. (2001) Greedy function approximation: A gradient boosting
machine.  Annals of Statistics.  Volume 29, Number 5.  1189 -- 1232.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\examples{

x <- matrix(rnorm(5000), ncol = 10)
y <- rnorm(nrow(x), x[,1] > 0, .1)

## y is a step function of x[, 1].  The other 9 x's are irrelevant.

model <- BoomBart(y ~ x,
                  niter = 1000,
                  initial.number.of.trees = 1,
                  number.of.trees.prior = PoissonPrior(1, 1, 200))

BartPartialDependencePlot(model,
                          1,
                          data.fraction = .5,
                          thin = 25,
                          ylim = range(model$response))


# Notice:
colnames(model$design.matrix)[1]  ## "x1"

# So this is another way to produce the same plot:
BartPartialDependencePlot(model,
                          "x1",
                          data.fraction = .5,
                          thin = 25,
                          ylim = range(model$response))
}

\keyword{hplot}
\keyword{models}
\keyword{tree}
\keyword{nonparametric}
