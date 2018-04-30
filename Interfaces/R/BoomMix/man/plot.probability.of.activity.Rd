\name{plot.probability.of.activity}
\alias{PlotProbabilityOfActivity}
\title{Plot the probability a process is active}
\Rdversion{1.0}

\description{ Plots the probability a component process of the MMPP is
  active. }

\usage{
PlotProbabilityOfActivity(mmpp,
                          process,
                          which.point.process = 1,
                          from = NULL,
                          to = NULL,
                          xlab = "Time",
                          ylab = "Probability of Activity",
                          activity.color = "lightblue",
                          ...)
}

\arguments{
  \item{mmpp}{An object of class
    \code{\link{MarkovModulatedPoissonProcess}}.}

  \item{process}{The name of a component process in \code{mmpp}.}

  \item{which.point.process}{The index (name or number) of a particular
    PointProcess object in training data for \code{mmpp}. }

  \item{from}{An object that can be coerced to \code{\link{POSIXt}}.
    The starting point for the plotting window.}

  \item{to}{An object that can be coerced to \code{\link{POSIXt}}.
    The ending point for the plotting window.}

  \item{xlab}{Label for the horizontal axis.}

  \item{ylab}{Label for the vertical axis.}

  \item{activity.color}{The color to shade the region where the process
    is active.}

  \item{\dots}{Extra arguments passed to \code{\link{plot}}.}

}

\value{
  This function is called for its side effect, which is to produce a
  plot on the current graphics device.
}

\seealso{
  \code{\link{MarkovModulatedPoissonProcess}},
  \code{\link{PoissonProcessComponent}},
  \code{\link{PoissonProcess}}
}

\examples{
data(syslog.timestamps)
spec <- list(background = PoissonProcessComponent(
               process = HomogeneousPoissonProcess(GammaPrior(1.0, 1.0))),
             user.birth = PoissonProcessComponent(
               process = HomogeneousPoissonProcess(
                 GammaPrior(a = 10.0, prior.mean = 1.0)),
               spawns = c(
                 "user.traffic",
                 "user.death"),
               kills = "user.birth"),
             user.traffic = PoissonProcessComponent(
               process = HomogeneousPoissonProcess(GammaPrior(1.0, 1.0))),
             user.death = PoissonProcessComponent(
               process = HomogeneousPoissonProcess(GammaPrior(1.0, 1.0)),
               spawns = "user.birth",
               kills = c(
                 "user.death",
                 "user.traffic")))

model <- MarkovModulatedPoissonProcess(
             syslog.timestamps,
             spec,
             initial.state = c("background", "user.birth"),
             niter = 1000)
par(mfrow = c(2,1))
PlotProbabilityOfActivity(model, "user.traffic", main = "Full Data")
PlotProbabilityOfActivity(model,
                          "user.traffic",
                          from = "2012-10-25",
                          to = "2012-10-26",
                          main = "October 25")

}
