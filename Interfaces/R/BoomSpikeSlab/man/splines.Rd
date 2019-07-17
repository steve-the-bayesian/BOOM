% Copyright 2019 Steven L. Scott. All Rights Reserved.
% Author: steve.the.bayesian@gmail.com (Steve Scott)

\name{spliunes}
\Rdversion{1.1}
\alias{BsplineBasis}
\alias{MsplineBasis}
\alias{IsplineBasis}
\alias{knots.SplineBasis}

\title{
  Spline Basis Expansions
}

\description{

}
\usage{
  BsplineBasis(x, knots = NULL, numknots = 3)
  MsplineBasis(x, knots = NULL, numknots = 3)
  IsplineBasis(x, knots = NULL, numknots = 3)

  \method{knots}{SplineBasis}(Fn, ...)

}

\arguments{

  \item{x}{
    A numeric vector to be expanded.
  }

  \item{knots}{ A numeric vector of knots defining the expansion.  The
    smallest and largest elements in \code{knots} defines the range of
    the expansion.  These knots are (notionally) replicated infinitely
    many times.  }

  \item{numknots}{If the knot vector is \code{NULL} then create a vector
    of length \code{numknots} that partitions \code{x} into
    \code{numknots} + 1 eqiprobable segments.}

  \item{Fn}{A spline basis matrix.}

  \item{\dots}{Unused, but required to match the signature of the
    \code{knots} generic function in the \code{stats} package.}
}

\details{
  B-splines are the basis most commonly used for additive
  regression models.

  M-splines are an alternative to B-splines, but are rarely used.

  I-splines are integrated M-splines.  These are monotonic functions,
  which is useful in monotonic regression problems.  If all regression
  coefficients are positive then the resulting function is
  nondecreasing.
}

\value{
  \code{XsplineBasis} returns a matrix formed by the spline basis
  expansion of \code{x}.

  \code{knots(Fn)} returns the \code{knots} attribute of \code{Fn},
  which might be useful in a second call to the basis expansion
  function.
}

\references{
  deBoor, "A Practical Guide to Splines".

  George and McCulloch (1997), "Approaches to Bayesian Variable
  Selection", \emph{Statistica Sinica}, \bold{7}, 339 -- 373.
  \url{http://www3.stat.sinica.edu.tw/statistica/oldpdf/A7n26.pdf}

  Ghosh and Clyde (2011) "Rao-Blackwellization for Bayesian variable
  selection and model averaging in linear and binary regression: A novel
  data augmentation approach", \emph{Journal of the American Statistical
  Association}, \bold{106} 1041-1052.
  \url{http://homepage.stat.uiowa.edu/~jghsh/ghosh_clyde_2011_jasa.pdf}
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{SpikeSlabPrior}},
  \code{\link{plot.lm.spike}},
  \code{\link{summary.lm.spike}},
  \code{\link{predict.lm.spike}}.
}

\examples{
  n <- 100
  p <- 10
  ngood <- 3
  niter <- 1000
  sigma <- .8

  x <- cbind(1, matrix(rnorm(n * (p-1)), nrow=n))
  beta <- c(rnorm(ngood), rep(0, p - ngood))
  y <- rnorm(n, x \%*\% beta, sigma)
  x <- x[,-1]
  model <- lm.spike(y ~ x, niter=niter)
  plot.ts(model$beta)
  hist(model$sigma)  ## should be near 8
  plot(model)
  summary(model)
  plot(model, "residuals")

  ## Now replace the first observation with a big outlier.
  y[1] <- 50
  model <- lm.spike(y ~ x, niter = niter)
  model2 <- lm.spike(y ~ x, niter = niter, error.distribution = "student")
  pred <- predict(model, newdata = x)
  pred2 <- predict(model2, newdata = x)

  ## Maximize the plot window before making these box plots.  They show
  ## the posterior predictive distribution of all 100 data points, so
  ## make sure your screen is 100 boxes wide!
  par(mfrow = c(2,1))
  BoxplotTrue(t(pred), truth = y, ylim = range(pred), pch = ".",
     main = "Posterior predictive distribution assuming Gaussian errors.")
  BoxplotTrue(t(pred2), truth = y, ylim  = range(pred), pch = ",",
     main = "Posterior predictive distribution assuming Student errors.")

  ## The posterior predictive distributions are much tighter in the
  ## student case than in the Gaussian case, even though the student
  ## model has heavier tails, because the "sigma" parameter is smaller.
  par(mfrow = c(1,1))
  CompareDensities(list(gaussian = model$sigma, student = model2$sigma),
                        xlab = "sigma")
}
\keyword{models}
\keyword{regression}
