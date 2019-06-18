\name{model.matrix}
\Rdversion{1.1}
\alias{GetPredictorMatrix}
\title{
  GetPredictorMatrix
}
\description{
  Extract the matrix of predictors.
}

\usage{
  GetPredictorMatrix(object, newdata, na.action = na.omit, ...)
}

\arguments{
  \item{object}{
    An object of class glm.spike.  The object must be a
    list with the following elements
    \itemize{
      \item beta: a matrix of MCMC draws, with rows representing draws,
      and columns representing coefficients.

      \item xlevels: the levels of any contrasts present in the original
      training data.

      \item contrasts: the "contrasts" attribute of the original design
      matrix used to train the model.

      \item terms: the terms of the formula used to fit the original model.
    }
  }

  \item{newdata}{A data frame, matrix, or vector containing the
       predictors needed to make a prediction.  If newdata is a
       matrix it must have the same number of columns as
       length(object$beta), unless it is off by one and the model
       contains an intercept, in which case an intercept term will
       be added.  If length(object$beta) == 1 (or 2, with one
       element containing an intercept) then newdata can be a
       numeric vector.
     }

  \item{na.action}{A function specifying what to do with \code{NA}'s.}

  \item{\dots}{
    Extra arguments passed to \code{\link{model.matrix}}, in the event
    that \code{newdata} is a data frame.
  }

}

\value{ A matrix of predictor variables suitable for multiplication by
     \code{object$beta}.  }

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}}
  \code{\link{plot.lm.spike}}
  \code{\link{predict.lm.spike}}
}

