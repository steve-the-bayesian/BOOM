


\name{markov.modulated.poisson.process}
\alias{MarkovModulatedPoissonProcess}
\title{Markov modulated Poisson process}
\Rdversion{1.0}

\description{
  Fits a Markov modulated Poisson process (MMPP), with or
  without marks attached to each event.
}

\usage{
MarkovModulatedPoissonProcess(point.process.list,
                              process.specification,
                              initial.state,
                              mixture.components = NULL,
                              known.source = NULL,
                              niter,
                              ping = niter / 10,
                              seed = NULL)
}

\arguments{
  \item{point.process.list}{An object of class
    \code{\link{PointProcess}}, or a list of such objects, giving the
    time points to be modeled.}

  \item{process.specification}{A list containing one or more
    \code{\link{PoissonProcessComponent}} objects specifying the latent
    processes, the processes spawned and killed by each type of event,
    and the mixture components that model any associated marks.}

  \item{initial.state}{A character vector containing a subset of the
    names from \code{process.specification} defining any valid state in
    the MMPP.  The model will use this information to determine all the
    other states.}

  \item{mixture.components}{A list of mixture components created using
    constructors from the \code{\link{BoomMix-package}} package.  These
    model the marks associated with the timestamps in
    \code{point.process.list}.  If \code{NULL} then no marks will be
    modeled.}

  \item{known.source}{A list of character vectors with the same
    dimensions as \code{point.process.list}.  Each vector contains
    zero or more strings naming the processes or mixture components
    that could have been the source for each event.  The elements of
    a \code{known.source} vector are names that can correspond to
    either the names of individual point processes in
    \code{process.specification}, or the names of mark models
    identified in \code{process.specification}.

    \itemize{

      \item{If the names match one or more processess in
        \code{process.specification} then those processes are assumed to
        be the only candidates that could have produced the
        corresponding event.}

      \item{If the names match one or more mark models from
        \code{process.specification} then the processes associated
        with those models are assumed to be the only candidates
        that could have produced the corresponding event.}

      \item{If the vector of names is empty (\code{character(0)}) then
        it is assumed that any of the processes in
        \code{process.specification} could have produced the event, in
        the sense that no processes are disallowed.}

      \item{If the entire \code{known.source} object is \code{NULL} then
        there are no processes that should be disallowed as potential
        sources for any of the points.}
    }

    As a convenience, if the point.process.list contains a single
    \code{\link{PointProcess}}, then \code{known.source} can be given as
    a character vector.  In this case each entry of \code{known.source}
    identifies a single \code{\link{PointProcess}} or mark model
    associated with a particular time point.

  }

  \item{niter}{The desired number of MCMC iterations.}

  \item{ping}{ The frequency of status update messages.  If \code{ping >
    0} then the program will print a status message to the screen every
    \code{ping} MCMC iterations.}

  \item{seed}{An integer to use as the random seed for the underlying
    C++ code.  If \code{NULL} then the seed will be set using the
    clock.}  }

\value{ An object of class \code{MarkovModulatedPoissonProcess}, which
  is a list with the following values.  }

\references{
  Fruhwirth-Schnatter (2006), "Finite mixture and Markov switching
  models", Springer.

  Scott and Smyth (2002), "The Markov Moduldated Poisson Process and the
  Markov Poisson Cascade". \emph{Bayesian Statistics 7}.  Oxford
  University Press.  pp 671 -- 680.
}

\author{
  Steven L. Scott \email{steve.the.bayesian@gmail.com}
}

\seealso{
  \code{\link{BoomMix-package}}.
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
\keyword{models}
