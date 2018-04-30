


\alias{BoomEvents-package}
\name{boom.events.package}
\title{Point process modeling}
\Rdversion{1.0}

\description{A package for fitting various flavors of Markov modulated
  Poisson processes using MCMC or (in the future) the EM algorithm.}

\section{Overview}{

  A point process is a data structure composed of a sequence of events.
  This package is concerned with temporal point processes, where events
  occur in sequence and are marked with time stamps.  Each event may
  also carry a set of measurements known as marks.  This package
  contains tools for building probabilistic models of point process
  data.

  The fundamental building block for point process models is the Poisson
  process.  A Poisson process is a stochastic model that assumes the
  number of events in an interval \eqn{I}{I} is a Poisson random
  variable with mean \deqn{\int_I\lambda(t) \ dt .}{ integral_I \lambda(t) dt.}

  The function \eqn{\lambda(t)}{lambda(t)} is known as the rate
  function.  If it is constant then the Poisson process is
  \emph{homogeneous}, which is the elementary case taught in
  undergraduate courses.

  If the rate function varies over time the process is
  \emph{inhomogeneous}.  A modeler who works with an inhomogeneous
  Poisson process must specify the form of the rate function, which must
  be a a pure function of time (and not past process behavior).  One as
  an example of an inhomogeneous Poisson process is a
  \code{\link{WeeklyCyclePoissonProcess}}, where the instantaneous rate
  is a function of the day of the week and hour of the day.
}

\section{Beyond the Poisson process}{

  The Poisson process is closely related to the continuous time Markov
  process.  It is possible to represent a continuous time Markov process
  with \eqn{S}{S} states as S Poisson processes "racing" to produce the
  first event.  The first Poisson process (other than the one associated
  with the current state) to produce an event determines the new state,
  and the time of the state transition is the time of the first event.
  Each state can have a different set of S 'component' processes.

  The relationship between the Poisson and Markov processes can be used
  to model non-Poisson data by mixing together several Poisson processes
  in a model known as the Markov modulated Poisson process (MMPP).  The
  MMPP assumes an unobserved Markov process determines the value of
  \eqn{\lambda(t)}{lambda(t)} at time t.

  One convenient way to parameterize an MMPP is to assume a set of
  Poisson component processes that are activated and deactivated by a
  latent Markov process that is represented by a pair of "birth" and
  "death" Poisson processes associated with each state.  Each component
  process produces events independently when it is "on," and its rate
  drops to zero when it is "off".

  It is convenient to assume that transitions in the latent Markov
  process also produce events (see
  \url{http://stevethebayesian.com/ScottSmythV7.pdf}), although this
  assumption is not absolutely necessary
  (\url{http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2006.00566.x/abstract}).
  If the "birth" and "death" processes also produce events, then the
  latent Markov process cannot switch states between two observed
  events, which means that the model can be viewed as a parameterized
  discrete time hidden Markov model.  Standard forward-backward
  recursions for hidden Markov models can be used to track the
  probabilistic state of the hidden Markov process and most other
  interesting probabilities related to the model.

  One useful class of MMPP is the Poisson cluster process.  The Poisson
  cluster process has three component processes.  A background process
  produces occasional rare events.  A primary process, when active,
  produces a set of 'primary' events marking the beginning of a new
  traffic burst.  Each primary event activates the secondary process,
  which runs for a short period of time at high intensity until the
  first event produced by the associated death process.

}

\section{Incorporating marks}{

  The MMPP can be used to classify events in a point process as coming
  from one of the component processes.  When used as a classifier, it is
  helpful to consider the distribution of the marks (if any) associated
  with each time point.  Let \eqn{y_t}{y[t]} be the mark associated with
  event t, and let each process in the MMPP be associated with a
  probability (or density) function capable of measuring
  \eqn{y_t}{y[t]}.

  With this generative model, it is easy to use the marks to help
  classify events.  The mark distributions operate just like a finite
  mixture model where the 'mixing weights' are proportional to the rate
  functions of the active processes at the time of the event.

  Mark distributions can be specified using code from the
  \code{\link{BoomMix-package}}.
}

\section{Tools for data analysis}{

  This package also contains some tools for visualizing and modeling
  point processes.  There is a plot method for
  \code{\link{PointProcess}} data objects.  As well as tools for
  computing the \code{\link{AverageEventRate}} and the
  \code{\link{AverageExposureTime}} for a data set of
  \code{PointProcess}-es.

}

\references{
  Scott and Smyth (2003)  "The Markov Modulated Poisson Process and
  Markov Poisson Cascade with Applications to Web Traffic Data",
  \url{http://stevethebayesian.com/ScottSmythV7.pdf}.

  Fearnhead and Sherlock (2006) "An exact Gibbs sampler for the
  Markov-modulated Poisson process".
  \url{http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2006.00566.x/abstract}.


}
