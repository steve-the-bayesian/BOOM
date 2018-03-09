\name{max.window.width}
\alias{MaxWindowWidth}
\alias{MaxWindowWidth.default}
\alias{MaxWindowWidth.DateRangeHoliday}
\title{Maximum Window Width for a Holiday}
\Rdversion{1.1}

\description{The maximum width of a holiday's influence window}

\usage{
\method{MaxWindowWidth}{default}(holiday, ...)
\method{MaxWindowWidth}{DateRangeHoliday}(holiday, ...)
}

\arguments{

  \item{holiday}{An object of class \code{\link{Holiday}}.}

  \item{\dots}{Other arguments (not used).}
}


\value{
  Returns the number of days in a holiday's influence window.
}


\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{Holiday}}.
  \code{\link{AddRegressionHoliday}}.
  \code{\link{AddRandomWalkHoliday}}.
  \code{\link{AddHierarchicalRegressionHoliday}}.
}

\examples{

easter <- NamedHoliday("Easter", days.before = 2, days.after = 1)
if (MaxWindowWidth(easter) == 4) {
  print("That's the right answer!\n")
}

## This holiday lasts two days longer in 2005 than in 2004.
may18 <- DateRangeHoliday("May18",
     start = as.Date(c("2004-05-17",
                       "2005-05-16")),
     end   = as.Date(c("2004-05-19",
                       "2005-05-20")))

if (MaxWindowWidth(may18) == 5) {
   print("Right again!\n")
}

}
\keyword{models}
\keyword{regression}
