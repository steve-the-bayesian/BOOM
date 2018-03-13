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
                                 time0, 
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
  stopifnot(inherits(holiday, "holiday"))
  ans <- list(holiday = holiday,
              time0 = .SetTimeZero(time0, y),
              sigma.prior = .ValidateHolidaySigmaPrior(sigma.prior, sdy),
              initial.state.prior = .ValidateHolidayInitialStatePrior(
                initial.state.prior, sdy))
  class(ans) <- c("RandomWalkHolidayStateModel", "StateModel")
  return(ans)
}
                                 
