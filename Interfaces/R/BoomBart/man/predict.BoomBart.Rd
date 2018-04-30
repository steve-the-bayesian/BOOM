\name{predict.BoomBart}
\title{predict.BoomBart}
\description{predict method for Bart models}
\alias{predict.BoomBart}

\usage{
\method{predict}{BoomBart}(
   object,
   newdata,
   distribution = c("function", "prediction"),
   burn = SuggestBartBurn(object),
   thin = 10,
   scale = c("trees", "data"),
   mean.only = FALSE,
   ...)
}

\arguments{
  \item{object}{A BoomBart model on which to base the predictions.}

  \item{newdata}{A data.frame containing the variables used in
    object$formula.}

  \item{distribution}{A string indicating the type of posterior
    predictive distribution desired.  If "function" then the predictive
    distribution of function values at 'newdata' is returned.  If
    "prediction" then the posterior predictive distribution of a new
    data value is returned.}

  \item{burn}{The number of MCMC iterations to discard as burn-in.}

  \item{thin}{The frequency of MCMC iterations to keep, after burn-in.
    E.g. if thin = 10 then every 10th draw will be used.}

  \item{scale}{The scale on which the predictions are to be made.  If
    this is is "trees" then they predicted values are on the
    link-function scale.  E.g. on the logit or probit scale for binary
    data, or the log scale for Poisson data.  If 'scale' is "data" then
    the inverse link function is applied.}

  \item{mean.only}{Logical.  If \code{TRUE} then the function returns
    the posterior mean.  Otherwise it returns the full posterior
    distribution.}

  \item{...}{Extra arguments are not used.  This argument is here to
    comply with the signatrue of the default S3 predict method.}

}

\value{
  If \code{mean.only} is \code{TRUE} then the return value is a vector
  of lenth \code{nrow(newdata)} giving the posterior mean of the
  predicted value for each observation (row) in \code{newdata}.

  If \code{mean.only} is \code{FALSE} then the return value is a matrix
  with rows corresponding to MCMC draws of posterior function values and
  columns corresponding to observations in \code{newdata}.
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

pred <- predict(model, x[1:5, ])
## pred contains the posterior distribution for the function values at
## the first two rows of x.
}
