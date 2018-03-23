.ValidateHolidaySigmaPrior <- function(sigma.prior, sdy) {
  ## Check that 'sigma.prior' is of class SdPrior, or NULL.  If NULL then return
  ## a default prior expressing the belief that sigma is small relative to sdy.
  ## The support of the prior is [0, sdy].
  ##
  ## The 'sigma' in question here refers to how much the same day of a holiday
  ## can vary from one instance to the next.  It does not describe the
  ## differences between successive days of a holiday influence window.
  ##
  ## Args:
  ##   sigma.prior: An object of class SdPrior, or NULL, representing the prior
  ##     for sigma.
  ##   sdy:  The sample standard deviation of the target series.
  ##
  ## Returns:
  ##   If sigma.prior is of class SdPrior then it is returned unchanged.  If it
  ##   is NULL then a default SdPrior is returned instead.  Any other case is an
  ##   error.
  if (is.null(sigma.prior)) {
    return(SdPrior(.01 * sdy, upper.limit = sdy))
  }
  stopifnot(inherits(sigma.prior, "SdPrior"))
  return(sigma.prior)
}

.ValidateHolidayInitialStatePrior <- function(initial.state.prior, sdy) {
  ## Args:
  ##   initial.state.prior: An object of class NormalPrior representing the
  ##     prior belief about variation among individual days in a holiday
  ##     influence window.  This argument can also be NULL.
  ##   sdy: The standard deviation of the target series, used to specify a
  ##     default prior in the case where the first argument is NULL.
  ##
  ## Returns:
  ##   If initial.state.prior is a NormalPrior then it is returned unchanged.
  ##   If it is NULL then a default NormalPrior is returned instead.  Any other
  ##   case is an error.
  if (is.null(initial.state.prior)) {
    return(NormalPrior(0, sdy))
  }
  stopifnot(inherits(initial.state.prior, "NormalPrior"))
  return(initial.state.prior)
}

AddRandomWalkHoliday <- function(state.specification = NULL,
                                 y,
                                 holiday,
                                 time0 = NULL, 
                                 sigma.prior = NULL,
                                 initial.state.prior = NULL,
                                 sdy = sd(as.numeric(y), na.rm = TRUE)) {
  ## Adds a random walk holiday state model to the state specification.
  ## This model says
  ## 
  ##    y[t] = (other state) + alpha[d(t), t] + observation_error, 
  ##
  ## where there is one element in alpha for each day in the holiday influence
  ## window.  The transition equation is
  ##
  ##   alpha[d(t+1), t+1] = alpha[d(t+1), t] + state_error
  ##
  ## if t+1 occurs on day d(t+1) of the influence window, and
  ##
  ##   alpha[d(t+1), t+1] = alpha[d(t+1), t]
  ##
  ## otherwise.
  ##
  ## Args:
  ##   state.specification: An existing list of state components.  The random
  ##     walk holiday component will be added to this list, and the augmented
  ##     list will be returned.
  ##   y:  The time series being modeled.  
  ##   time0: an object coercible to class Date giving the date of the first
  ##     observation in the time series.
  ##   holiday: An object inheriting from class Holiday, describing the
  ##     beginning and end of the influence window for the holiday modeled by
  ##     this state component.
  ##   sigma.prior: An object of class SdPrior giving the prior distribution of
  ##     the state innovation variance.
  ##   initial.state.prior: An object of class NormalPrior describing how much
  ##     influence a typical holiday might be expected to have.
  ##   sdy: The standard deviation of the time series being modeled.  This is
  ##     used to set initial.state.prior or sigma.prior in the event they are
  ##     passed as NULL.  If both priors are supplied by the user, or if 'y' is
  ##     supplied, then this argument is unneccessary.
  ##
  ## Returns:
  ##   A list containing the information in the arguments, formatted as expected
  ##   by the underlying C++ code.
  if (is.null(state.specification)) state.specification <- list()
  stopifnot(is.list(state.specification))
  stopifnot(inherits(holiday, "Holiday"))
  holiday.model <- list(
    name = holiday$name,
    holiday = holiday,
    time0 = as.Date(.SetTimeZero(time0, y)),
    sigma.prior = .ValidateHolidaySigmaPrior(sigma.prior, sdy),
    initial.state.prior = .ValidateHolidayInitialStatePrior(
      initial.state.prior, sdy))
  class(holiday.model) <- c("RandomWalkHolidayStateModel", "HolidayStateModel",
    "StateModel")
  state.specification[[length(state.specification) + 1]] <- holiday.model
  return(state.specification)
}

plot.RandomWalkHolidayStateModel <- function(x,
                                             bsts.object,
                                             burn = NULL,
                                             time = NULL,
                                             style = NULL,
                                             ylim = NULL,
                                             ...) {
  ## S3 plot method for RandomWalkHolidayStateModel
  ## Args:
  ##   x:  An object of type RandomWalkHolidayStateModel.
  ##   bsts.object: A bsts model that includes a RandomWalkHolidayStateModel
  ##     component.
  ##   burn:  The number of MCMC iterations to discard.
  ##   time:  Not used.  Here for compatibility with plot.StateModel.
  ##   style:  Not used.  Here for compatibility with plot.StateModel.
  ##   ylim:  Limits for the vertical axis of the plot.
  ##   ...: Extra arguments passed to boxplot.
  ##
  ## Side Effects:
  ##   A plot is added to the current graphics device.  Side-by-side boxplots
  ##   show the evolution of the holiday effect across time for each instance of
  ##   the holiday.  Each group of boxes shows the pattern for the holiday
  ##   window.  The is one such grouping for each distinct holiday window in the
  ##   training data for the bsts.object.
  ##
  ## Returns:
  ##   invisible(NULL)
  state.specification <- x
  stopifnot(inherits(state.specification, "RandomWalkHolidayStateModel"))
  stopifnot(inherits(bsts.object, "bsts"))

  state <- bsts.object$state.contributions[, state.specification$name, ]
  if (is.null(state)) {
    stop("Could not find state contributions for ", state.specification$name)
  }
  if (!is.matrix(state)) {
    state <- matrix(state, ncol = 1)
  }
  if (is.null(burn)) {
    burn <- 0
  }
  if (burn > 0) {
    state <- state[-(1:burn), ]
  }

  if (is.null(ylim)) {
    ylim <- range(state)
  }
  all.zero <- apply(state == 0, 2, all)

  index <- seq_along(all.zero)
  index <- index[!all.zero]
  gaps <- 1 + which(diff(index) > 1)

  old.par <- par(mar = c(2, 2, 4, 2))
  on.exit(par(old.par))
  
  boxplot(.NonzeroCols(state), main = state.specification$name, pch = 20,
    cex = .2, axes = FALSE, ylim = ylim, ...)
  box()
  axis(2)
  abline(v = gaps - .5, lty = 3)
  abline(h = 0, lty = 3)
  return(invisible(NULL))
}
                                             
