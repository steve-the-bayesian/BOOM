\name{model.matrix.glm.spike}
\Rdversion{1.1}
\alias{model.matrix.glm.spike}
\title{
  Construct Design Matrices
}
\description{
  Creates a matrix of predictors appropriate for glm.spike models.
}

\usage{
  \method{model.matrix}{glm.spike}(object, \dots)
}
\arguments{
  \item{object}{
    An object of class \code{glm.spike}.
  }

  \item{\dots}{
    Extra arguments passed to \code{\link{model.matrix.default}}.
  }

}

\value{
     The matrix of predictors used at training time, so long as the
     original data used to fit the model is available in the frame
     where this function is called.
}

\details{
  \code{glm.spike} objects do not store the predictors used to fit the
  model.  If the training data is modified between when \code{object}
  is fit and when this function is called, the modifications will
  be reflected in the returned value.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
}

\keyword{models}
\keyword{regression}
