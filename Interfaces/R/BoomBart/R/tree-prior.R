BartTreePrior <- function(total.prediction.sd,
                          root.split.probability = .95,
                          split.decay.rate = 2,
                          number.of.trees.prior =
                              DiscreteUniformPrior(1, 200)) {
  ## A prior distribution to be passed to BoomBart.
  ##   total.prediction.sd: The amount by which you expect the
  ##     predictions to vary, on the "link scale" (i.e. the raw scale
  ##     for Gaussian models, the probit scale for probits, the logit
  ##     scale for logits, and the log scale for Poisson).  This
  ##     parameter determines the prior for the mean parameters at the
  ##     leaves of the trees, which is N(0, total.prediction.sd^2 /
  ##     number.of.trees)
  ##   root.split.probability, split.decay.rate: These two work
  ##     together to determine the prior over tree topology.  The
  ##     probability of a split at a node at depth d (the root is at d
  ##     = 0) is alpha / (1 + d)^beta.  The probability of a split at
  ##     the root is simply alpha (so alpha = root.split.probability),
  ##     and beta controls the rate at which the split probability
  ##     decays as you move to greater depths.  For small, but
  ##     non-trivial trees set alpha slightly less than one and beta
  ##     to 2 or more.
  ##   number.of.trees.prior: An object of class DiscretePrior giving
  ##     a prior distribution over the number of trees.
  ##
  ## Returns:
  ##   An object of class BartTreePrior.
  stopifnot(inherits(number.of.trees.prior, "DiscretePrior"))
  stopifnot(root.split.probability > 0 && root.split.probability <= 1)
  stopifnot(split.decay.rate > 0)
  stopifnot(total.prediction.sd > 0)

  ans <- list(total.prediction.sd = total.prediction.sd,
              root.split.probability = root.split.probability,
              split.decay.rate = split.decay.rate,
              number.of.trees.prior = number.of.trees.prior)
  class(ans) <- c("BartTreePrior", "Prior")
  return(ans)
}

##======================================================================
GaussianBartTreePrior <- function(
    total.prediction.sd,
    root.split.probability = .95,
    split.decay.rate = 2,
    number.of.trees.prior = DiscreteUniformPrior(1, 200),
    sigma.guess = sqrt(sdy^2 * (1 - expected.r2)),
    sigma.weight = 3,
    expected.r2 = .5,
    sdy) {
  ## A prior for a Gaussian Bart model.  This is identical to
  ## BartTreePrior, but with extra arguments for specifying a prior
  ## distribution for the residual standard deviation parameter sigma.
  ## Args:
  ##   total.prediction.sd,
  ##   root.split.probability,
  ##   split.decay.rate,
  ##   number.of.trees.prior:  See documentation in BartTreePrior
  ##   sigma.guess:  A prior guess at the value of sigma.
  ##   sigma.weight: The number of observations worth of weight to
  ##     give to sigma.guess.
  ##   expected.r2, sdy: These are ignored if sigma.guess is specified
  ##     directly.  Otherwise, sdy is the standard deviation of the
  ##     response variable, and expected.r2 is the expected R^2 in the
  ##     non-parametric regression of y on the X's.
  ##
  ## Returns:
  ##   An object of class GaussianBartTreePrior.
  ans <- BartTreePrior(total.prediction.sd,
                       root.split.probability,
                       split.decay.rate,
                       number.of.trees.prior)
  ans$sigma.prior <- SdPrior(sigma.guess, sigma.weight)
  class(ans) <- c("GaussianBartTreePrior", class(ans))
  return(ans)
}
