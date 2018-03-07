\name{mlm.spike}
\Rdversion{1.1}
\alias{mlm.spike}
\title{
  Spike and slab multinomial logistic regression
}
\description{

  MCMC algorithm for multinomial logist models with a 'spike-and-slab'
  prior that places some amount of posterior probability at zero for a
  subset of the regression coefficients.

}
\usage{
mlm.spike(subject.formula,
          choice.formula = NULL,
          niter,
          data,
          choice.name.separator = ".",
          contrasts = NULL,
          subset,
          prior = NULL,
          ping = niter / 10,
          proposal.df = 3,
          rwm.scale.factor = 1,
          nthreads = 1,
          mh.chunk.size = 10,
          proposal.weights = c("DA" = .5, "RWM" = .25, "TIM" = .25),
          seed = NULL,
          ...)
}
\arguments{

  \item{subject.formula}{A model \code{\link{formula}} describing the
    relationship between the response (which must be a factor) and the
    characteristics of the subjects associated with the decision
    process.  If there are no subject-level predictors then \code{y ~ 1}
    will provide a model with a different intercept for each level of
    the response.  If no intercepts are desired, use \code{y ~ 0}.  }

  \item{choice.formula}{A model \code{\link{formula}} describing the
    relationship between the response and the characteristics of the
    object being chosen.  This can be left \code{NULL} if no
    choice-level characteristics are to be used in the model.  The
    variables appearing on the right hand side must be stored in
    \code{data} with the name of response levels appended, and a
    chararacter (\code{choice.name.separator}) used as a separator.  For
    example, if "MPG" is one of the variables in the formula, and the
    response can assume values of "Toyota", "Honda", and "Chevy", then
    \code{data} must contain \code{MPG.Toyota}, \code{MPG.Honda}, and
    \code{MPG.Chevy}.}

  \item{niter}{ The number of MCMC iterations to run.  Be sure to
    include enough so you can discard a burn-in set.  }

  \item{data}{A data frame containing the data referenced in
    \code{subject.formula} and \code{choice.formula} arguments.  If
    \code{choice.formula} is \code{NULL} then this argument is optional,
    and variables will be pulled from the parent environment if it is
    omitted.  If \code{choice.formula} is non-\code{NULL}, then
    \code{data} must be supplied.  Each row in \code{data} represents a
    single observation containing the relevant data about both the
    subject making the choice, as well as about the items being chosen
    among.  A variable measuring a choice characteristic must be present
    for each choice level in the response variable.  The stems for the
    choice-variable names that measure the same concepts must be
    identical, and choice level must be appended as a suffix, separated
    by a "."  character.  Thus, if 'HP' is a variable to be considered,
    and the response levels are 'Toyota', 'Honda', 'Chevy', then the
    data must contain variables named 'HP.Toyota', 'HP.Honda', and
    'HP.Chevy'.}

  \item{choice.name.separator}{The character used to separate the
    predictor names from the choice values for the choice-level
    predictor variables in 'data'.}

  \item{contrasts}{ An optional list. See the \code{contrasts.arg} of
    \code{\link{model.matrix.default}}.  }

  \item{subset}{ An optional vector specifying a subset of observations
    to be used in the fitting process.  }

  \item{prior}{ An object of class
    \code{\link{IndependentSpikeSlabPrior}}.  The portions of the prior
    distribution relating to the residual variance are not used.

    A convenience function: \code{\link{MultinomialLogitSpikeSlabPrior}}
    is provided to help with the accounting headaches of vectorizing the
    \code{subject.beta} and \code{choice.beta} parameters.  }

  \item{ping}{The frequency with which status updates are printed to the
    console.  Measured in MCMC iterations.  If \code{ping < 0} then no
    status updates will be printed.}

  \item{proposal.df}{The "degrees of freedom" parameter that the
    Metropolis-Hastings algorithm should use for the multivariate T
    proposal distribution.  If \code{proposal.df <= 0} then a Gaussian
    proposal is used instead.}

  \item{rwm.scale.factor}{The scale factor to use for random walk
    Metropolis updates.  See details.}

  \item{nthreads}{The number of CPU-threads to use for data
    augmentation.}

  \item{mh.chunk.size}{The maximum number of coefficients to draw in a
    single "chunk" of a Metropolis-Hastings update.  See details. }

  \item{proposal.weights}{A vector of 3 probabilities (summing to 1)
    indicating the probability of each type of MH proposal during each
    iteration.  The weights should be given names "DA", "RWM", and
    "TIM" for clarity.}

  \item{seed}{Seed to use for the C++ random number generator.  It
    should be \code{NULL} or an int.  If \code{NULL} the seed value will
    be taken from the global \code{\link{.Random.seed}} object.}

  \item{\dots}{
    Extra arguments to be passed to \code{\link{MultinomialLogitSpikeSlabPrior}}.
  }

}
\value{

  Returns an object of class \code{mlm.spike}, which inherits from
  \code{logit.spike} and \code{lm.spike}.  The returned object is a list
  with the following elements

  \item{beta}{A \code{niter} by \code{ncol(x)} matrix of regression
    coefficients, many of which may be zero.  Each row corresponds to an
    MCMC iteration.}

  \item{prior}{The prior used to fit the model.  If a \code{prior} was
    supplied as an argument it will be returned.  Otherwise this will be
    the automatically generated prior based on the other function
    arguments. }

  \item{MH.accounting}{A summary of the amount of time spent in each
    type of MCMC move, and the acceptance rate for each move type. }

}

\details{
  \subsection{Model Details:}{
   A multinomial logit model has two sets of predictors: one measuring
   characterisitcs of the subject making the choice, and the other
   measuring characteristics of the items being chosen.  The model
   can be written

   \deqn{ Pr(y[i] = m) \propto exp(beta.subject[, m] * x.subject[i, ]
     + beta.choice * x.choice[i, , m])}{
     Pr(y[i] = m) (proportional to)
     exp(beta.subject[, m] * x.subject[i, ] + beta.choice * x.choice[i, , m])}

   The coefficients in this model are beta.subject and beta.choice.
   beta.choice is a subject.xdim by ('nchoices' - 1) matrix.  Each row
   multiplies the design matrix produced by subject.formula for a
   particular choice level, where the first choice level is omitted
   (logically set to zero) for identifiability.  beta.choice is a
   vector multiplying the design matrix produced by choice.formula,
   and thre are 'nchoices' of such matrices.

   The coefficient vector 'beta' is the concatenation
   c(beta.subject, beta.choice), where beta.subject is vectorized
   by stacking its columns (in the usual R fashion).  This means
   that the first contiguous region of beta contains the
   subject-level coefficients for choice level 2.
  }

  \subsection{MCMC Details:}{
   The MCMC algorithm randomly moves between three tyes of
   updates: data augmentation, random walk Metropolis (RWM), and
   tailored independence Metropolis (TIM).
   \itemize{
    \item  DA: Each observation in the model is
      associated with a set of latent variables that renders the
      complete data posterior distribution conditionally Gaussian.
      The augmentation scheme is described in Tuchler (2008).  The
      data augmentation algorithm conditions on the latent data,
      and integrates out the coefficients, to sample the inclusion
      vector (i.e. the vector of indicators showing which
      coefficients are nonzero) using Gibbs sampling.  Then the
      coefficients are sampled given complete data conditional on
      inclusion.  This is the only move that attemps a dimension
      change.

    \item RWM: A chunk of the coefficient vector (up to mh.chunk.size)
      is selected.  The proposal distribution is either
      multivariate normal or multivariate T (depending on
      'proposal.df') centered on current values of this chunk.
      The precision parameter of the normal (or T) is the negative
      Hessian of the un-normalized log posterior, evaluated at the
      current value.  The precision is divided by
      rwm.scale.factor.  Only coefficients currently included in
      the model at the time of the proposal will be modified.

    \item TIM: A chunk of the coefficient vector (up to mh.chunk.size)
      is selected.  The proposal distribution is constructed by
      locating the posterior mode (using the current value as a
      starting point).  The proposal is a Gaussian (or
      multivariate T) centered on the posterior mode, with
      precision equal to the negative Hessian evaluated at the
      mode.  This is an expensive, but effective step.  If the
      posterior mode finding fails (for numerical reasons) then a
      RWM proposal will be attempted instead.
    }
  }
}
\references{
  Tuchler (2008), "Bayesian Variable Selection for Logistic Models Using
  Auxiliary Mixture Sampling", \emph{Journal of Computational and
  Graphical Statistics}, \bold{17} 76 -- 94.
}

\author{
  Steven L. Scott
}

\seealso{
  \code{\link{lm.spike}}
  \code{\link{SpikeSlabPrior}},
  \code{\link{plot.lm.spike}},
  \code{\link{summary.lm.spike}},
  \code{\link{predict.lm.spike}}.
}

\examples{

rmulti <- function (prob) {
  ## Sample from heterogeneous multinomial distributions.
    if (is.vector(prob)) {
        S <- length(prob)
        return(sample(1:S, size = 1, prob = prob))
    }
    nc <- apply(prob, 1, sum)
    n <- nrow(prob)
    S <- ncol(prob)
    u <- runif(n, 0, nc)
    alive <- rep(TRUE, n)
    z <- numeric(n)
    p <- rep(0, n)
    for (s in 1:S) {
        p <- p + prob[, s]
        indx <- alive & (u < p)
        alive[indx] <- FALSE
        z[indx] <- s
        if (!any(alive))
            break
    }
    return(z)
}

## Define sizes for the problem
subject.predictor.dimension <- 3
choice.predictor.dimension <- 4
nchoices <- 5
nobs <- 1000

## The response can be "a", "b", "c", ...
choice.levels <- letters[1:nchoices]

## Create "subject level characteristics".
subject.x <- matrix(rnorm(nobs * (subject.predictor.dimension - 1)),
                    nrow = nobs)
subject.beta <- cbind(
    0, matrix(rnorm(subject.predictor.dimension * (nchoices - 1)),
              ncol = nchoices - 1))
colnames(subject.x) <- state.name[1:ncol(subject.x)]

## Create "choice level characteristics".
choice.x <- matrix(rnorm(nchoices * choice.predictor.dimension * nobs),
                   nrow = nobs)
choice.characteristics <- c("foo", "bar", "baz", "qux")
choice.names <- as.character(outer(choice.characteristics, choice.levels, FUN = paste, sep = ":"))
colnames(choice.x) <- choice.names
choice.beta <- rnorm(choice.predictor.dimension)

## Combine an intercept term, subject data, and choice data.
X <- cbind(1, subject.x, choice.x)
p <- ncol(X)
true.beta <- c(subject.beta[, -1], choice.beta)
Beta <- matrix(nrow = nchoices, ncol = p)
for (m in 1:nchoices) {
  Beta[m, ] <- rep(0, p)
  Beta[m, 1:subject.predictor.dimension] <- subject.beta[, m]
  begin <- subject.predictor.dimension + 1 + (m-1) * choice.predictor.dimension
  end <- begin + choice.predictor.dimension - 1
  Beta[m, begin:end] <- choice.beta
}

eta <- X \%*\% t(Beta)
prob <- exp(eta)
prob <- prob / rowSums(prob)
response <- as.factor(choice.levels[rmulti(prob)])
simulated.data <- as.data.frame(X[, -1])
simulated.data$response <- response

model <- mlm.spike(response ~ Alabama + Alaska,
                   response ~ foo + bar + baz + qux,
                   niter = 500,
                   choice.name.separator = ":",
                   expected.subject.model.size = -1,
                   expected.choice.model.size = -1,
                   data = simulated.data,
                   proposal.weights = c("DA" = .8, "RWM" = .1, "TIM" = .1))

}

\keyword{models}
\keyword{regression}
